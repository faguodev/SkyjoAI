import functools
import json
import logging
import random

import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from stable_baselines3 import PPO

from callback_functions import SkyjoLogging_and_SelfPlayCallbacks
from custom_models.action_mask_model import TorchActionMaskModel
from environment.skyjo_env import env as skyjo_env
from custom_models.fixed_policies import RandomPolicy
from environment.skyjo_game import SkyjoGame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global_turn_count = 0

#region Ours
# Load configuration
model_path = "obs_simple_indirect_True_vf_True_cr_5_ar_1_decay_0.98_ent_0.03_nn_[256, 256]"
checkpoint = "checkpoint_8500"
step = "self_play"
config_path = f"logs/{step}/{model_path}/experiment_config.json"

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
    #.env_runners(num_env_runners=5)
    #.rollouts(num_rollout_workers=5, num_envs_per_worker=1)
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

model_save_dir = f"trained_models/{step}/{model_path}"
final_dir = model_save_dir + f"/{checkpoint}"

algo.restore(final_dir)
#endregion

#region Theirs
##############################
# LOAD SB3 MODEL
##############################
# Suppose you have a model saved at "PPO_1M_multi.zip"
sb3_model_path = "custom_models/PPO_1M_multi.zip"
model_env2 = PPO.load(sb3_model_path)

##############################
# OBSERVATION MAPPING
##############################
def obs_one_to_obs_two(obs_one: dict, turn_counter: int) -> np.ndarray:
    """
    Convert Environment One observation dict into an Environment Two–style observation.

    For a 2-player game in Environment One (PettingZoo):
      obs_one["observations"] might have length 43:
        - Indices:
          3 -> discard_top
          4 -> deck_card
          19..31 -> 12 cards for player0
          31..43 -> 12 cards for player1

    We’ll build a single array of length 27:
      [discard_top, turn_counter, deck_card, 12 board vals (player0), 12 board vals (player1)]

    Hidden/refunded cards = 15 or -14 => we turn them into 5.0 for environment Two.
    """
    obs_vec = obs_one["observations"]
    discard_top = obs_vec[17]
    deck_card   = obs_vec[18]

    # Grab the 12 cards for player0
    raw_board_p0 = obs_vec[19:31]
    board_int_p0 = []
    for val in raw_board_p0:
        if val == 15:
            board_int_p0.append(5.0)
        
        elif val == -14:  # Hidden/refunded
            board_int_p0.append(0.0)
        else:
            board_int_p0.append(float(val))

    # Grab the 12 cards for player1
    raw_board_p1 = obs_vec[31:43]
    board_int_p1 = []
    for val in raw_board_p1:
        if val == 15 or val == -14:
            board_int_p1.append(5.0)
        else:
            board_int_p1.append(float(val))

    state = float(turn_counter)

    obs_env2 = np.concatenate((
        np.array([discard_top, state, deck_card], dtype=np.float32),
        np.array(board_int_p0, dtype=np.float32),
        np.array(board_int_p1, dtype=np.float32),
    ))

    #print("obs_env")
    #print(obs_env2)

    return obs_env2


##############################
# ACTION MAPPING
##############################
def act_two_to_act_one(action_two: np.ndarray, expected_action: str, obs_one: dict) -> int:
    """
    Convert the (2,) action from environment Two to a single integer action for environment One.

    action_two[0]: 0=draw from deck, 1=take from discard
    action_two[1]: 0..11=place at position, 12=discard & reveal

    Environment One action mapping:
    - expected_action="draw": 
        if draw_or_take=0 -> 24 (draw from deck)
        if draw_or_take=1 -> 25 (take from discard)
    - expected_action="place":
        if place_or_discover<12 -> place that position (0..11)
        if place_or_discover=12 -> reveal a random masked card:
            valid reveal actions are 12..23, filtered by action_mask.
    """
    draw_or_take = action_two[0]
    place_or_discover = action_two[1]

    if expected_action == "draw":
        global global_turn_count
        global_turn_count += 1
        # Draw step
        if draw_or_take == 0:
            return 24  # draw from draw pile
        else:
            return 25  # take from discard pile

    elif expected_action == "place":

        # Place step
        if place_or_discover < 12:
            # place hand card onto that position
            return place_or_discover
        else:
            # Need to reveal a card: pick a random masked card action from 12..23
            action_mask = obs_one["action_mask"]
            reveal_actions = [a for a in range(12, 24) if action_mask[a] == 1]
            if len(reveal_actions) == 0:
                # fallback if none available (should not happen)
                return random.choice([a for a in range(0, 12) if action_mask[a] == 1])
            else:
                return random.choice(reveal_actions)

    # fallback
    return 24


##############################
# POLICY FUNCTION
##############################
def sb3_policy_env2(obs_one: dict, game) -> int:
    """
    Policy function integrating the SB3 model from environment Two into environment One.

    Steps:
       1) Convert PettingZoo obs -> env2 obs (with turn count)
       2) model.predict(obs_env2)
       3) convert env2 action -> env1 action
    """
    global global_turn_count

    obs_env2 = obs_one_to_obs_two(obs_one, global_turn_count)
    action_two, _ = model_env2.predict(obs_env2, deterministic=True)

    next_action = game.get_expected_action()[1]  # "draw" or "place"
    action_one = act_two_to_act_one(action_two, next_action, obs_one)
    return action_one

#endregion

def random_admissible_policy(observation, action_mask):
    """picks randomly an admissible action from the action mask"""
    return np.random.choice(
        np.arange(len(action_mask)),
        p= action_mask/np.sum(action_mask)
    )

def pre_programmed_smart_policy_simple(obs):
    observation = obs["observations"]
    action_mask = obs["action_mask"]
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
            if observation[idx_start] == 5:
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

    all_good = True
    game = SkyjoGame(
        num_players=2, 
        observe_other_player_indirect=False,
        observation_mode="simple"    
    )
    i_episode += 1
    global_turn_count = 0
    game_over = False

    while not game_over:
        
        # Get current state
        agent = game.get_expected_action()[0]

        try:
            obs, mask = game.collect_observation(agent)
        except Exception as e:
            print(f"Exception occurred on observation collection for agent {agent}")
            print(e)
            all_good = False
            break
        
        # Get action from appropriate model
        if agent == 0:

            if observe_other_player_indirect:
                observation = {
                    "observations": obs[:31],
                    "action_mask": mask
                }
            else:
                observation = {
                    "observations": obs,
                    "action_mask": mask
                }
            # action = pre_programmed_smart_policy(observation)
            
            policy = algo.get_policy(policy_id=policy_mapping_fn(agent, None, None))
            action_exploration_policy, _, action_info = policy.compute_single_action(observation)
            # 
            action = action_exploration_policy
        else:
            observation = {
                "observations": obs,
                "action_mask": mask
            }
            action = sb3_policy_env2(observation, game)

            # action = pre_programmed_smart_policy(obs, mask)
            
        # Execute action
        try:
            game_over, _ = game.act(agent, action)
        except Exception as e:
            print(f"Exception occurred on move from agent {agent}")
            print(e)
            all_good = False
            wins_dict[1-agent] += 1
            break

    if game.has_terminated and all_good:
        scores = [int(x) for x in game.game_metrics["final_score"]]
        print(f"Scores: {scores}")
        (
            stats_counts,
            cards_sum,
            n_hidden,
            top_discard,
        ) = game._jit_observe_global_game_stats(
            game.players_cards,
            game.players_masked,
            np.array(game.discard_pile, dtype=game.players_cards.dtype),
            count_players_cards=not game.observe_other_player_indirect,
        )
        print(n_hidden)
        print(cards_sum)
        winner = scores.index(min(scores))
        wins_dict[winner] += 1
    else:
        print("There was some issue")
    

print(wins_dict)