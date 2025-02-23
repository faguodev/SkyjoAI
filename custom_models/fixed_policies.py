import random
import numpy as np
from ray.rllib.policy.policy import Policy
from stable_baselines3 import PPO

class RandomPolicy(Policy):
    def compute_actions(
        self,
        obs,
        state_batches,
        prev_action_batch,
        prev_reward_batch,
        info_batch,
        episodes,
        explore,
        timestep,
        **kwargs,
    ):
        
        action_mask = obs[0][:26] #["action_mask"]
        observation = obs[0][26:] #["observations"]

        admissible_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        action = random.choice(admissible_actions)
        return np.asarray([action]), [], {}

    def learn_on_batch(self, samples):
        return {}  # No learning, as this is a fixed policy.

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

class PreProgrammedPolicySimple(Policy):
    def compute_actions(
        self,
        obs,
        state_batches,
        prev_action_batch,
        prev_reward_batch,
        info_batch,
        episodes,
        explore,
        timestep,
        **kwargs,
    ):
        action_mask = obs[0][:26] #["action_mask"]
        observation = obs[0][26:] #["observations"]
        action = 26
        #print(observation[19:])
        #print(observation)
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
        return np.asarray([action]), [], {}
    def learn_on_batch(self, samples):
        return {}  # No learning, as this is a fixed policy.
    def get_weights(self):
        return {}
    def set_weights(self, weights):
        pass

class PreProgrammedPolicyOneHot(Policy):
    def compute_actions(
        self,
        obs,
        state_batches,
        prev_action_batch,
        prev_reward_batch,
        info_batch,
        episodes,
        explore,
        timestep,
        **kwargs,
    ):
        action_mask = obs[0][:26] #["action_mask"]
        observation = obs[0][26:] #["observations"]

        action = 26
        admissible_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        #check wether card still has to be taken from discard pile or draw pile
        if 24 in admissible_actions:
            #choose wether to take the upper most card from the discard pile or choose a random from the draw pile
            #if card on discard pile has value smaller equal 3 take it
            if 1 in observation[17:23]:
                action = 25
            #else draw random card from draw pile
            else:
                action = 24
        #if card was already taken from deck/discard pile continue with placing/throwing away
        else:
            #go through one-hot-encoded hand cards to find value of hand card
            for i, hand in enumerate(observation[34:50]):
                if hand == 1:
                    hand_card_value = i - 2
            #find position and highest value of players cards (here unknown cards are valued as 5)
            max_card_value = -2
            masked_cards = []
            for i in range(12):
                idx_start = i*17+51
                #print("starting idx:",idx_start)
                #print("observation looping through: ", observation[idx_start:idx_start+16])
                #find value of current card (17th one-hot-encoded field is for refunded cards and therefore ignored)
                for j, val in enumerate(observation[idx_start:idx_start+16]):
                    if val == 1:
                        if j == 15:
                            masked_cards.append(i)
                            #print(f"appending {i}")
                            if max_card_value < 5:
                                max_card_value = 5
                                imax = i
                        elif max_card_value < j - 2:
                            max_card_value = j-2
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
        return np.asarray([action]), [], {}
    def learn_on_batch(self, samples):
        return {}  # No learning, as this is a fixed policy.
    def get_weights(self):
        return {}
    def set_weights(self, weights):
        pass

class SingleAgentPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.sb3_model_path = "custom_models/PPO_1M_multi.zip"
        self.model_env2 = PPO.load(self.sb3_model_path)


    def compute_actions(
        self,
        obs,
        state_batches,
        prev_action_batch,
        prev_reward_batch,
        info_batch,
        episodes,
        explore,
        timestep,
        **kwargs,
    ):
        
        """
        Policy function integrating the SB3 model from environment Two into environment One.

        Steps:
        1) Convert PettingZoo obs -> env2 obs (with turn count)
        2) model.predict(obs_env2)
        3) convert env2 action -> env1 action
        """
        action_mask = obs[0][:26] #["action_mask"]
        observation = obs[0][26:] #["observations"]

        #print(type(observation))

        obs_env2 = self._obs_one_to_obs_two(observation)
        action_two, _ = self.model_env2.predict(obs_env2, deterministic=True)

        next_action = observation[1]  # "draw" or "place"
        action_one = self._act_two_to_act_one(action_two, next_action, action_mask)

        #print(action_one)
        return np.asarray([action_one]), [], {}

    ##############################
    # OBSERVATION MAPPING
    ##############################
    def _obs_one_to_obs_two(self, obs_one: np.ndarray) -> np.ndarray:
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
        turn_counter = obs_one[0]
        obs_vec = obs_one[2:]
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
    def _act_two_to_act_one(self, action_two: np.ndarray, expected_action: str, action_mask) -> int:
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

        if expected_action == 0:
            # Draw step
            if draw_or_take == 0:
                return 24  # draw from draw pile
            else:
                return 25  # take from discard pile

        elif expected_action == 1:

            # Place step
            if place_or_discover < 12:
                # place hand card onto that position

                if action_mask[place_or_discover]:
                    return place_or_discover
                else:
                    print("################################\n\nCatastrophic Failure0")
                    return random.choice([a for a in range(0, 12) if action_mask[a] == 1])
            else:
                # Need to reveal a card: pick a random masked card action from 12..23
                reveal_actions = [a for a in range(12, 24) if action_mask[a] == 1]
                if len(reveal_actions) == 0:
                    # fallback if none available (should not happen)
                    print("################################\n\nCatastrophic Failure1")
                    return random.choice([a for a in range(0, 12) if action_mask[a] == 1])
                else:
                    action = random.choice(reveal_actions)
                    if not action_mask[action]:
                        print("################################\n\nCatastrophic Failure2")
                        return random.choice([a for a in range(0, 12) if action_mask[a] == 1])
                    return action

        # fallback
        return 24
    


class PreProgrammedPolicyEfficientOneHot(Policy):
    def compute_actions(
        self,
        obs,
        state_batches,
        prev_action_batch,
        prev_reward_batch,
        info_batch,
        episodes,
        explore,
        timestep,
        **kwargs,
    ):
        
        
        action_mask = obs[0][:26] #["action_mask"]
        observation = obs[0][26:] #["observations"]
        action = 26
        #print(observation[19:])
        #print(observation)
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
        return np.asarray([action]), [], {}
    def learn_on_batch(self, samples):
        return {}  # No learning, as this is a fixed policy.
    def get_weights(self):
        return {}
    def set_weights(self, weights):
        pass