import functools
import glob
import json
import logging
import os
from typing import Optional
import random

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
    model_path = "obs_efficient_one_hot_port_to_other_indirect_False_vf_True_cr_2_ar_1_decay_0.995_ent_0.01_nn_[64, 64]_against_other2"
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
    
    def preprogrammed_policy_port_to_other(
        observation,
        action_mask
    ):
        observation = observation[2:]
        #print(observation)
        action = 26
        admissible_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        #check wether card still has to be taken from discard pile or draw pile
        if 24 in admissible_actions:
            #choose wether to take the upper most card from the discard pile or choose a random from the draw pile
            #if card on discard pile has value smaller equal 3 take it
            if observation[17] <= 3:
                action = 25
            #else draw random card from draw pile
            else:
                action = 24
        #if card was already taken from deck/discard pile continue with placing/throwing away
        else:
            #go through one-hot-encoded hand cards to find value of hand card
            # for i, hand in enumerate(observation[34:50]):
            #     if hand == 1:
            hand_card_value = observation[18]
            #find position and highest value of players cards (here unknown cards are valued as 5)
            max_card_value = -2
            masked_cards = []
            for i in range(12):
                idx_start = i+19
                #print("starting idx:",idx_start)
                #print("observation looping through: ", observation[idx_start:idx_start+16])
                #find value of current card (17th one-hot-encoded field is for refunded cards and therefore ignored)
                if observation[idx_start+12] == 1:
                    assert observation[idx_start] == 5, ("One hot unknown does not relate to true unknown")
                    masked_cards.append(i)
                    #print(f"appending {i}")
                    if max_card_value < 5:
                        max_card_value = 5
                        imax = i
                elif max_card_value < observation[idx_start]:
                    max_card_value = observation[idx_start]
                    imax = i
            #print(masked_cards)
            #1st case hand card value is lower equal than 3 (if card was taken from discard this branch will be taken for 100%)
            #place card on position with max_card_value
            if hand_card_value <= 3:
                action = imax
            #else if hand is smaller than max_card_value replace card with higest value with handcard
            elif hand_card_value < max_card_value:
                action = imax
            #else throw hand card away and reveal masked card
            else:
                action = 12 + random.choice(masked_cards)
                for a in admissible_actions:
                    if a in np.array(masked_cards) + 12:
                        action = a
                #print(action, "chosen action - observation: ", observation[(51 + (action-12)*17):(68 + (action-12)*17)])
            assert action != 26, ("No Valid action was chosen!")
        return action

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
                #action = random_admissible_policy(observation, action_mask)
                action = preprogrammed_policy(observation, action_mask)
            # elif agent == 2:
            #     policy = algo.get_policy(policy_id=policy_mapping_fn(1, None, None))
            #     action_exploration_policy, _, action_info = policy.compute_single_action(obs)
            #     # 
            #     action = action_exploration_policy

                #action = random_admissible_policy(observation, action_mask)

            #print(f"{action_mask = }")
            #print(f"sampled action {agent}: {action}")
            env_pettingzoo.step(action)
            if term:
                env_pettingzoo.step(None)
                #print('done', reward)
                final_scores = env_pettingzoo.table.get_game_metrics()["final_score"]
                #print(env_pettingzoo.table.get_game_metrics())

                winner = np.argmin(final_scores)

                #print("========================")
                #print("AND THE WINNER IS")
                #print(winner)
                wins_dict[winner] = wins_dict[winner] + 1
                break
        if i_episode % 500 == 0:
            print(wins_dict)

    print(wins_dict)


