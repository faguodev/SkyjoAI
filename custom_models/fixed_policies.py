import random
import numpy as np
from ray.rllib.policy.policy import Policy

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