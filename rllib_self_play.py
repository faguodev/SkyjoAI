from ray import tune, train, init, shutdown
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
import numpy as np
import os
import json

from models.action_mask_model import TorchActionMaskModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#max number of policy updates (can take multiple timesteps per iteration and train on this mini-batch)
max_iterations = 100
#max number of timesteps taken in game
max_timesteps = 1000

max_league_size = 5


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

        self.win_rate_threshold = win_rate_threshold

    def on_episode_end(self, *, worker, metrics_logger, base_env, policies, episode, **kwargs):
        """
        This is called at the end of each episode. We grab
        the final card sum from the `info` dict for each agent
        and log it as a custom metric.
        """
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info is not None and "final_card_sum" in info:
                metric_name = f"final_card_sum_{agent_id}"
                episode.custom_metrics[metric_name] = info["final_card_sum"]
            if info is not None and "n_hidden_cards" in info:
                metric_name = f"n_hidden_cards_{agent_id}"
                episode.custom_metrics[metric_name] = info["n_hidden_cards"]

        main_agent = episode.episode_id % 3
        rewards = episode.get_rewards()

        if main_agent in rewards:
            #True if main won False if not id of main is episode_id%3 (definition) id of winner is first
            main_won = np.argmin(base_env.table.get_game_metrics()["final_score"]) == episode.episode_id % 3 #rewards[main_agent][-1] == 1.0
            metrics_logger.log_value(
                "win_rate",
                main_won,
                window=100,
            )

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):

        win_rate = result[ENV_RUNNER_RESULTS]["win_rate"]
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
                return (
                    "main"
                    if episode.episode_id % 3 == agent_id
                    else "main_v{}".format(
                        np.random.choice(list(range(1, self.current_opponent + 1)))
                    )
                )

            main_policy = algorithm.get_policy("main")
            new_policy = algorithm.add_policy(
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
        result["league_size"] = self.current_opponent + 2

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
    # "fcnet_hiddens": [1024, 1024, 1024, 512, 512],
    "fcnet_activation": "tanh",
}

param_space = {
    "lr": tune.grid_search([0.0001, 0.001, 0.01]),  # Learning rate options
    "model": tune.grid_search([{"custom_model": TorchActionMaskModel, "fcnet_activation": "relu"}, {"custom_model": TorchActionMaskModel, "fcnet_activation": "tanh"}])
}

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
    return "main" if episode.episode_id % 3 == agent_id else "random"

config = (
    PPOConfig()
    .training()#model=model_config, )
    .environment("skyjo", env_config=skyjo_config)
    .framework('torch')
    .callbacks(functools.partial(
        SkyjoLogging_and_SelfPlayCallbacks,
        win_rate_threshold=0.85,
        )
    )
    #.callbacks(RewardDecayCallback)
    .env_runners(num_env_runners=5)
    .rollouts(num_rollout_workers=20, num_envs_per_worker=1)
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

#define stopping conditions for Self_play
#Num_env_steps_samples_lifetime... max number of in game steps sampled
#training_iteration... max num of updates to policy (max number of training steps)
#"league_size"... min size of league (starts with size 3), therefore main is logged "league_size" -3 times
stop = {
    NUM_ENV_STEPS_SAMPLED_LIFETIME: max_timesteps,
    TRAINING_ITERATION: max_iterations,
    "league_size": max_league_size,
}

storage_path = os.path.join(os.getcwd(), "results")

tuner = tune.Tuner(
    trainable="PPO",
    param_space={**config.to_dict(), **param_space},
    run_config=train.RunConfig(
        stop=stop,
        storage_path=storage_path,
        checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=1,
                checkpoint_at_end=True,
            ),
    ),
    #tune_config=tune.TuneConfig(
            #num_samples=args.num_samples,
            #max_concurrent_trials=args.max_concurrent_trials,
            #scheduler=scheduler,
    #),
)
result = tuner.fit()