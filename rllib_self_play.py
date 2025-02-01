from ray import tune, train, init, shutdown
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
#from ray.tune.stopper import MaximumIterationStopper
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)


from environment.skyjo_env import env as skyjo_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import functools
import logging
import os
import json

from models.action_mask_model import TorchActionMaskModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RewardDecay_Callback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Decay the reward scaling factor over training iterations
        action_reward_decay = max(0.05, 1.0 - result["training_iteration"] * 0.005)
        # env = algorithm.workers.local_worker().env
        # env = algorithm.workers.local_env_runner.env
        algorithm.config.env_config["action_reward_decay"] = action_reward_decay
        logger.info(action_reward_decay)


class SkyjoLogging_and_SelfPlayCallbacks(DefaultCallbacks):

    def __init__(self, win_rate_threshold):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
        self.winning_policy = []

        self.win_rate_threshold = win_rate_threshold


    def on_episode_end(        
        self,
        *,
        episode,
        **kwargs,): #policies, 
        """
        This is called at the end of each episode. We grab
        the final card sum from the `info` dict for each agent
        and log it as a custom metric.
        """
        
        #win_id = 0
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
    
            if info is not None and "final_card_sum" in info:
                metric_name = f"final_card_sum_{agent_id}"
                episode.custom_metrics[metric_name] = info["final_card_sum"]
            if info is not None and "n_hidden_cards" in info:
                metric_name = f"n_hidden_cards_{agent_id}"
                episode.custom_metrics[metric_name] = info["n_hidden_cards"]

                #episode.custom_metrics["winning_policy"].append([episode.policy_for(id) for id in info["winner_ids"]])
                
    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger,
        result,
        **kwargs,
        ):

        #for p_id in self.playing_polices:
        #    print("Policies playing: ", p_id)

        
        #opponent_rew_2 = result[ENV_RUNNER_RESULTS]["hist_stats"].pop(f"policy_{self.playing_polices[2]}_reward")
        main_rew = result[ENV_RUNNER_RESULTS]["hist_stats"].pop("policy_main_reward")
        won = 0
        n_games = len(main_rew)
        for rew in main_rew:
            if rew%1 != 0:
                won += 1

        win_rate = won/ len(main_rew)

        # INFO: This only works with old rewards... 
        # For new rewards, one needs to figure out a better way to do this.
        # Unfortunately, the results lists will be of unequal length for the
        # different policies, as they are not participating in the same number
        # of games. One would need to figure out a way to reconstruct who played
        # against whom.

        #for r_main in main_rew:
        #    if r_main == 100.0:
        #        won += 1

        #win_rate = won / len(main_rew)

        
        print(f"Iter={algorithm.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > self.win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played modules
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # agent_id = [0|1] -> policy depends on episode ID
                # This way, we make sure that both policies sometimes play
                # (start player) and sometimes agent1 (player to move 2nd).
                if agent_id == 0:
                    return "main"

                else:
                    return "main_v{}".format(
                        np.random.choice(list(range(1, self.current_opponent + 1)))
                    )

            main_policy = algorithm.get_policy("main")
            new_policy = algorithm.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(main_policy),
                policy_mapping_fn=policy_mapping_fn,
                config=main_policy.config,
                observation_space=main_policy.observation_space,
                action_space=main_policy.action_space,
            )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            main_state = main_policy.get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            algorithm.env_runner_group.sync_weights()

        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 3

        #print(f"Matchups:\n{self._matching_stats}")

skyjo_config = {
    "num_players": 3,
    "score_penalty": 2.0,
    "observe_other_player_indirect": False,
    "mean_reward": 1.0,
    "reward_refunded": 10,
    "final_reward": 100,
    "score_per_unknown": 5.0,
    "action_reward_decay": 1.0,
    "old_reward": False,
    "render_mode": "human",
}

model_config = {
    "custom_model": TorchActionMaskModel,
    # Add the following keys:
    "fcnet_hiddens": [512, 512, 512, 512],
    "fcnet_activation": "relu",
}

# param_space = {
#     "lr": tune.grid_search([0.0001, 0.001, 0.01]),  # Learning rate options
#     "model": tune.grid_search([{"custom_model": TorchActionMaskModel, "fcnet_activation": "relu"}, {"custom_model": TorchActionMaskModel, "fcnet_activation": "tanh"}])
# }

def env_creator(config):
    return PettingZooEnv(skyjo_env(**config))

register_env("skyjo", env_creator)

test_env = env_creator(skyjo_config)
obs_space = test_env.observation_space
act_space = test_env.action_space

#def policy_mapping_fn(agent_id, _, **kwargs):
#return "policy_" + str(agent_id) #int(agent_id.split("_")[-1])

#policy mapping for self_play: only one "main" policy is trained
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if 0 == agent_id:
        return "main"
    elif 1 == agent_id:
        return "random_1"
    else:
        return "random_2"

config = (
    PPOConfig()
    .training(model=model_config, )
    .environment("skyjo", env_config=skyjo_config)
    .framework('torch')
    .callbacks(functools.partial(
        SkyjoLogging_and_SelfPlayCallbacks,
        win_rate_threshold=0.5,
        )
    )
    #.callbacks(RewardDecayCallback)
    .env_runners(num_env_runners=1)
    .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
    .resources(num_gpus=1)
    .multi_agent(
        policies={
            "main": (None, obs_space[0], act_space[0], {"entropy_coeff":0.03}),
            "random_1": (None, obs_space[1], act_space[1], {"entropy_coeff":0.03}),
            "random_2": (None, obs_space[2], act_space[2], {"entropy_coeff":0.03})
        },
        policy_mapping_fn=policy_mapping_fn,#(lambda agent_id, *args, **kwargs: agent_id),
        policies_to_train=["main"],
    )
    .evaluation(evaluation_num_env_runners=0)
    .debugging(log_level="INFO")
    .api_stack(
        enable_rl_module_and_learner=False,
        # enable_env_runner_and_connector_v2=True,
    )
    # .training()
    #     lr = ,
    # )
)


algo = config.build()

def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable types."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return "error"

# trained_models_<beta>_<beta>_<beta>_<callback>
config = "_others_indirect"
model_save_dir = "v4_trained_models_old_rewards" + config
os.makedirs(model_save_dir, exist_ok=True)
max_steps = 1e6
max_iters = 10000


if not os.path.exists("logs4"):
    os.mkdir("logs4")

for iters in range(max_iters):
    result = algo.train()

    # Can be adjusted as needed
    if iters % 100 == 0:
        with open(f"logs4/result_iteration_{iters}.json", "w") as f:
            json.dump(result, f, indent=4, default=convert_to_serializable)

    if iters % 100 == 0:
        checkpoint_dir = model_save_dir + f"/checkpoint_{iters}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        algo.save(checkpoint_dir)
    if result["timesteps_total"] >= max_steps:
        print(f"training done, because max_steps {max_steps} {result['timesteps_total']} reached")
        break
else:
    print(f"training done, because max_iters {max_iters} reached")

final_dir = model_save_dir + f"/final"
os.makedirs(final_dir, exist_ok=True)
algo.save(final_dir)

#max number of policy updates (can take multiple timesteps per iteration and train on this mini-batch)
#max_iterations = 100
#max number of timesteps taken in game
#max_timesteps = 10000
#max number of policies in the league
#max_league_size = 5
#define stopping conditions for Self_play
#Num_env_steps_samples_lifetime... max number of in game steps sampled
#training_iteration... max num of updates to policy (max number of training steps)
#"league_size"... min size of league (starts with size 3), therefore main is logged "league_size" -3 times
# stop = {
#     NUM_ENV_STEPS_SAMPLED_LIFETIME: max_timesteps,
#     TRAINING_ITERATION: max_iterations,
#     "league_size": max_league_size,
# }

# storage_path = os.path.join(os.getcwd(), "results")

# tuner = tune.Tuner(
#     trainable="PPO",
#     param_space={**config.to_dict(), **param_space},
#     run_config=train.RunConfig(
#         stop=stop,
#         storage_path=storage_path,
#         checkpoint_config=train.CheckpointConfig(
#                 checkpoint_frequency=1,
#                 checkpoint_at_end=True,
#             ),
#     ),
#     #tune_config=tune.TuneConfig(
#             #num_samples=args.num_samples,
#             #max_concurrent_trials=args.max_concurrent_trials,
#             #scheduler=scheduler,
#     #),
# )
# result = tuner.fit()

#algo = config.build()
#algo.train()