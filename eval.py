from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from environment.skyjo_env import env as skyjo_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import logging
import numpy as np
import os

from models.action_mask_model import TorchActionMaskModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RewardDecayCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Decay the reward scaling factor over training iterations
        action_reward_decay = max(0.05, 1.0 - result["training_iteration"] * 0.005)
        # env = algorithm.workers.local_worker().env
        # env = algorithm.workers.local_env_runner.env
        algorithm.config.env_config["action_reward_decay"] = action_reward_decay
        logger.info(action_reward_decay)


skyjo_config = {
    "num_players": 3,
    "score_penalty": 2.0,
    "observe_other_player_indirect": True,
    "mean_reward": 1.0,
    "reward_refunded": 10,
    "final_reward": 100,
    "score_per_unknown": 5.0,
    "action_reward_decay": 1.0,
    "render_mode": "human",
}

model_config = {
    'custom_model': TorchActionMaskModel 
}

def env_creator(config):
    return PettingZooEnv(skyjo_env(**config))

register_env("skyjo", env_creator)


test_env = env_creator(skyjo_config)
obs_space = test_env.observation_space
act_space = test_env.action_space

def policy_mapping_fn(agent_id, _, **kwargs):
    return "policy_" + str(agent_id) #int(agent_id.split("_")[-1])

config = (
    PPOConfig()
    .training(model=model_config, )
    .environment("skyjo", env_config=skyjo_config)
    .framework('torch')
    .callbacks(RewardDecayCallback)
    .env_runners(num_env_runners=1)
    # .rollouts(num_rollout_workers=6)
    .resources(num_gpus=1)
    .multi_agent(
        policies={
            "policy_0": (None, obs_space[0], act_space[0], {"entropy_coeff":0.01}),
            "policy_1": (None, obs_space[1], act_space[1], {"entropy_coeff":0.01}),
            "policy_2": (None, obs_space[2], act_space[2], {"entropy_coeff":0.01})
        },
        policy_mapping_fn=policy_mapping_fn,#(lambda agent_id, *args, **kwargs: agent_id),
    )
    .evaluation(evaluation_num_env_runners=0)
    .debugging(log_level="INFO")
)

algo = config.build()
algo_new = config.build()

model_save_dir = "trained_models_old_rewards"
model_save_dir_new = "trained_models"
final_dir = model_save_dir + f"/final"
final_dir_new = model_save_dir_new + f"/final"

algo.restore(final_dir)
algo_new.restore(final_dir_new)

env_pettingzoo = skyjo_env(**skyjo_config)
env_pettingzoo.reset()

def random_admissible_policy(observation, action_mask):
    """picks randomly an admissible action from the action mask"""
    return np.random.choice(
        np.arange(len(action_mask)),
        p= action_mask/np.sum(action_mask)
    )

wins_dict = {
    0: 0,
    1: 0,
    2: 0
}

i_episode = 1
while i_episode <= 10000:
    print("=============================")
    print(f"Iteration {i_episode}")
    i_episode += 1
    env_pettingzoo.reset()
    for i, agent in enumerate(env_pettingzoo.agent_iter(max_iter=6000)):
        # get observation (state) for current agent:
        #print(f"\n\n\n\n\n===================== Iteration {i} =====================")
        obs, reward, term, trunc, info = env_pettingzoo.last()

        #print(f"{term = }, {trunc = }")

        #print(env_pettingzoo.render())

        # store current state
        observation = obs["observations"]
        action_mask = obs["action_mask"]
        
        if agent == 0:
            action = random_admissible_policy(observation, action_mask)
        elif agent == 1:
            policy = algo.get_policy(policy_id=policy_mapping_fn(agent, None))
            action_exploration_policy, _, action_info = policy.compute_single_action(obs)
            # 
            action = action_exploration_policy
        elif agent == 2:
            policy = algo_new.get_policy(policy_id=policy_mapping_fn(agent, None))
            action_exploration_policy, _, action_info = policy.compute_single_action(obs)
            # 
            action = action_exploration_policy

        #print(f"{action_mask = }")
        #print(f"sampled action {agent}: {action}")
        env_pettingzoo.step(action)
        if term:
            env_pettingzoo.step(None)
            #print('done', reward)
            final_scores = env_pettingzoo.table.get_game_metrics()["final_score"]

            winner = np.argmin(final_scores)

            #print("========================")
            #print("AND THE WINNER IS")
            #print(winner)
            wins_dict[winner] = wins_dict[winner] + 1
            break

print(wins_dict)


