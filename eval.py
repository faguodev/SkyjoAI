import functools
import glob
import json
import logging
import os
from typing import Optional

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.policy import Policy
from ray.tune.registry import register_env

from callback_functions import (RewardDecay_Callback,
                                SkyjoLogging_and_SelfPlayCallbacks)
from custom_models.action_mask_model import TorchActionMaskModel
from custom_models.fixed_policies import RandomPolicy
from environment.skyjo_env import env as skyjo_env

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Load configuration
    model_path = "obs_simple_indirect_True_vf_True_cr_5_ar_1_decay_1_ent_0.01_nn_[2048, 2048, 1024, 512]"
    checkpoint = "checkpoint_1250"
    config_path = f"logs/grid_search/{model_path}/experiment_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract configuration parameters
    observation_mode = config["observation_mode"]
    observe_other_player_indirect = config["observe_other_player_indirect"]
    vf_share_layers = config["vf_share_layers"]
    curiosity_reward = config["curiosity_reward"]
    action_reward_reduction = config["action_reward_reduction"]
    action_reward_decay = config["action_reward_decay"]
    entropy_coeff = config["entropy_coeff"]
    neural_network_size = config["neural_network_size"]

    # Environment configuration
    skyjo_config = {
        "num_players": 2,
        "reward_config": {
            "score_penalty": 1.0,
            "reward_refunded": 10,
            "final_reward": 100,
            "score_per_unknown": 5.0,
            "action_reward_reduction": 0.0,
            "old_reward": False,
            "curiosity_reward": curiosity_reward,
        },
        "observe_other_player_indirect": observe_other_player_indirect,
        "render_mode": "human",
        "observation_mode": observation_mode,
    }

    # Model configuration
    model_config = {
        "custom_model": TorchActionMaskModel,
        "vf_share_layers": vf_share_layers,
        "fcnet_hiddens": neural_network_size,
        "fcnet_activation": "relu",
    }

    def env_creator(config):
        return PettingZooEnv(skyjo_env(**config))

    register_env("skyjo", env_creator)

    test_env = env_creator(skyjo_config)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "main" if agent_id == 0 else "policy_1"

    # Configure the algorithm
    config = (
        PPOConfig()
        .training(model=model_config)
        .environment("skyjo", env_config=skyjo_config)
        .framework("torch")
        .callbacks(
            functools.partial(
                SkyjoLogging_and_SelfPlayCallbacks,
                main_policy_id=0,
                win_rate_threshold=0.65,
                action_reward_reduction=action_reward_reduction,
                action_reward_decay=action_reward_decay,
            )
        )
        .env_runners(num_env_runners=5)
        .rollouts(num_rollout_workers=5, num_envs_per_worker=1)
        .resources(num_gpus=1)
        .multi_agent(
            policies={
                "main": (None, obs_space[0], act_space[0], {"entropy_coeff": entropy_coeff}),
                "policy_1": (RandomPolicy, obs_space[1], act_space[1], {"entropy_coeff": entropy_coeff}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
        )
        .evaluation(evaluation_num_env_runners=0)
        .debugging(log_level="INFO")
        .api_stack(enable_rl_module_and_learner=False)
        .learners(num_gpus_per_learner=1)
    )

    algo = config.build()

    model_save_dir = f"trained_models/grid_search/{model_path}"
    final_dir = model_save_dir + f"/{checkpoint}"

    algo.restore(final_dir)

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
    }

    i_episode = 1
    while i_episode <= 1000:
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
                policy = algo.get_policy(policy_id=policy_mapping_fn(agent, None, None))
                action_exploration_policy, _, action_info = policy.compute_single_action(obs)
                # 
                action = action_exploration_policy
            elif agent == 1:
                # 
                action = random_admissible_policy(observation, action_mask)
            elif agent == 2:
                policy = algo.get_policy(policy_id=policy_mapping_fn(1, None, None))
                action_exploration_policy, _, action_info = policy.compute_single_action(obs)
                # 
                action = action_exploration_policy

                #action = random_admissible_policy(observation, action_mask)

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
        if i_episode % 500 == 0:
            print(wins_dict)

    print(wins_dict)


