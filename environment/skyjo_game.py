import itertools
import math
import warnings
from typing import List, Tuple

import numpy as np

# use numba as optional dependency
try:
    from numba import njit
except ImportError:
    # in case numba is not installed -> njit functions are just python/slow.
    def njit(fastmath):
        def decorator(func):
            return func
        return decorator


class SkyjoGame(object):
    def __init__(
        self, 
        num_players: int = 3, 
        score_penalty=2, 
        observe_other_player_indirect=False,
        observation_mode="simple",
    ) -> None:
        """ """
        assert (
            0 < num_players <= 12
        ), "Skyjo can be played from 1 up to 8 (recommended) / 12 (theoretical) players"

        # init objects
        self.num_players = num_players
        self.score_penalty = score_penalty
        self.observation_mode = observation_mode

        self.global_turn_counter = 0

        # placeholders for unknown/refunded in the internal game logic:
        self.fill_masked_unknown_value = 15
        self.fill_masked_refunded_value = -14
        self.card_dtype = np.int8

        # action names for clarity
        self._name_draw = "draw"
        self._name_place = "place"

        # one-hot parameters
        #   => for each card "slot", we produce 17 entries:
        #      [-2..12] => 15 values, plus 1 unknown, plus 1 refunded
        if self.observation_mode == "simple" or self.observation_mode == "efficient_one_hot" or self.observation_mode == "simple_port_to_other" or self.observation_mode == "efficient_one_hot_port_to_other":
            self.one_hot_size = 1
        elif self.observation_mode == "onehot":
            self.one_hot_size = 12

        # observation of other players:
        self.observe_other_player_indirect = observe_other_player_indirect

        # >>> ADJUSTED OBSERVATION SHAPE <<<
        # The first 17 entries are still integer-based "global stats" (1 + 1 + 15).
        # Then we add 2 single-card fields (top_discard + hand_card), each 1/17 dims => +2/34
        # Finally, for the player's cards:
        #   - If indirect => only own 12 => 12*1/17 = 12/204
        #   - If direct   => all players => num_players*12*1/17
        #   => total = 17 + 2/34 + (12 or num_players*12)*1/17
        if self.observation_mode == "efficient_one_hot":
            if observe_other_player_indirect:
                self.obs_shape = (17 + 2 + 24,) # + additionally 12 one_hot info für verdeckte Karte = 1 wenn verdeckt 0 wenn offen.
            else:
                self.obs_shape = (17 + 2 + self.num_players*24,)
        elif self.observation_mode == "simple_port_to_other":
            if observe_other_player_indirect:
                self.obs_shape = (19 + 2*self.one_hot_size + 12*self.one_hot_size,)
            else:
                self.obs_shape = (19 + 2*self.one_hot_size + self.num_players*12*self.one_hot_size,)
        elif self.observation_mode == "efficient_one_hot_port_to_other":
            if observe_other_player_indirect:
                self.obs_shape = (19 + 2 + 24,)
            else:
                self.obs_shape = (19 + 2 + self.num_players*24,)

        else:    
            if observe_other_player_indirect:
                self.obs_shape = (17 + 2*self.one_hot_size + 12*self.one_hot_size,)
            else:
                self.obs_shape = (17 + 2*self.one_hot_size + self.num_players*12*self.one_hot_size,)

        self.action_mask_shape = (26,)
        self.previous_action = None

        self.has_terminated = False

        # reset
        self.reset()

    # [start: reset utils]
    def reset(self):
        # 150 cards from -2 to 12
        self.has_terminated = False
        # If None, no one has initiated the last round, else the player_id is saved to indicate who revealed all cards first
        self.last_round_initiator = None
        self.global_turn_counter = 0
        # metrics
        self.game_metrics = {
            "num_refunded": [0] * self.num_players,
            "num_placed": [0] * self.num_players,
            "final_score": False,
        }
        self.hand_card = self.fill_masked_unknown_value
        drawpile = self._new_drawpile(self.card_dtype)
        self.players_cards = drawpile[: 12 * self.num_players].reshape(
            self.num_players, -1
        )

        # discard_pile: first in last out
        self.drawpile, self.discard_pile = self._reshuffle_discard_pile(
            drawpile[self.num_players * 12 :]
        )

        self.players_masked = self._reset_card_mask(self.num_players, self.card_dtype)
        self._reset_start_player()
        assert self.expected_action[1] == self._name_draw, "expect to draw after reset"

    @staticmethod
    @njit(fastmath=True)
    def _new_drawpile(card_dtype=np.int8):
        """create a drawpile len(150) and cards from -2 to 12"""
        number_of_cards = [10 for _ in np.arange(-2, 13)]
        number_of_cards[0] = 5
        number_of_cards[2] = 15
        drawpile = np.repeat(np.arange(-2, 13, dtype=card_dtype), number_of_cards)
        np.random.shuffle(drawpile)
        return drawpile

    def set_seed(self, value):
        """adds a random number generator. does not affect global np.random.seed()"""
        self.rng = np.random.default_rng(value)
        self._set_seed_njit(value + 1)
        self.reset()

    @staticmethod
    @njit(fastmath=True)
    def _set_seed_njit(value: int):
        """set seed for numba"""
        np.random.seed(value)

    @staticmethod
    @njit(fastmath=True)
    def _reset_card_mask(num_players, card_dtype):
        players_masked = np.full((num_players, 12), 2, dtype=card_dtype)
        for pl in range(num_players):
            picked = np.random.choice(12, 2, replace=False)
            players_masked[pl][picked] = 1
        return players_masked

    def _reset_start_player(self):
        player_counts = self._jit_observe_global_game_stats(
            self.players_cards,
            self.players_masked,
            np.array(self.discard_pile, dtype=self.players_cards.dtype),
        )[1]
        # player with the largest sum of known cards starts
        starter_id = player_counts.argmax() * 2

        self.actions = itertools.cycle(
            (
                [player, action]
                for player in range(self.num_players)
                for action in [self._name_draw, self._name_place]
            )
        )

        # forward to the correct starting action
        for _ in range(1 + starter_id):
            self._internal_next_action()

    @staticmethod
    @njit(fastmath=True)
    def _reshuffle_discard_pile(old_pile) -> Tuple[List[int], List[int]]:
        """reshuffle discard pile into drawpile, pop top for discard pile"""
        np.random.shuffle(old_pile)
        drawpile = list(old_pile)
        discard_pile = list([drawpile.pop()])
        return drawpile, discard_pile

    # [end: reset utils]

    def _internal_next_action(self):
        """set next expected action"""
        self.expected_action = next(self.actions)

    # ------------------------------------------------------------------#
    #                           ONE-HOT UTILS                           #
    # ------------------------------------------------------------------#

    def _one_hot_card_value(self, card_value: int) -> np.ndarray:
        """
        Convert a single card_value (int) into a length-17 one-hot vector:
          indices 0..14 correspond to -2..12
          index 15 -> unknown (was 15 in the logic)
          index 16 -> refunded (was -14 in the logic)
        """
        arr = np.zeros(self.one_hot_size, dtype=np.int8)
        if -2 <= card_value <= 12:
            # shift by +2 to move [-2..12] -> [0..14]
            arr[card_value + 2] = 1
        elif card_value == self.fill_masked_unknown_value:
            arr[15] = 1
        elif card_value == self.fill_masked_refunded_value:
            arr[16] = 1
        else:
            # If you have any edge case, you might want to handle it or raise an error
            pass
        return arr

    def _one_hot_encode_array(self, card_array: List[int]) -> np.ndarray:
        """
        Convert a list/array of integer card values into flattened one-hot vectors.
        E.g. if card_array has length N -> output shape is (N * 17,)
        """
        out = []
        for val in card_array:
            out.append(self._one_hot_card_value(val))
        return np.concatenate(out, axis=0)

    # [start: collect observation]
    def collect_observation(self, player_id: int) -> Tuple[np.array, np.array]:
        # get global stats
        (
            stats_counts,
            cards_sum,
            n_hidden,
            top_discard,
        ) = self._jit_observe_global_game_stats(
            self.players_cards,
            self.players_masked,
            np.array(self.discard_pile, dtype=self.players_cards.dtype),
            count_players_cards=not self.observe_other_player_indirect,
        )

        # gather the known cards (still in integer form):
        if self.observe_other_player_indirect:
            #initialize one_hot array for unknown cards with correct size
            if self.observation_mode == "efficient_one_hot" or self.observation_mode == "efficient_one_hot_port_to_other":
                efficient_one_hot_obs = np.zeros(12)
            # observe only own 12 cards
            player_obs = self._jit_known_player_cards(
                self.players_cards,
                self.players_masked,
                fill_unknown=self.fill_masked_unknown_value,
                player_id=player_id,
            )
        else:
            #initialize one_hot array for unknown cards
            if self.observation_mode == "efficient_one_hot" or self.observation_mode == "efficient_one_hot_port_to_other":
                efficient_one_hot_obs = np.zeros(12 * self.num_players)

            # observe all players' cards
            player_obs = self._jit_known_player_cards_all(
                self.players_cards,
                self.players_masked,
                fill_unknown=self.fill_masked_unknown_value,
                player_id=player_id,
            )

        # ---- Build the numerical features for first part (17) ----
        #   [0] = min(cards_sum.min(), 127)
        #   [1] = n_hidden.min()
        #   [2..16] = stats_counts (length 15)
        #   => total = 17
        global_stats = np.array(
            ([min(cards_sum.min(), 127), n_hidden.min()] + stats_counts),
            dtype=self.card_dtype,
        )  # shape (17,)

        if self.observation_mode == "simple":

            player_obs = [5 if x == 15 else x for x in player_obs]

            obs = np.array(
                (
                    [min(cards_sum.min(), 127)]  # (1,)
                    + [n_hidden.min()]  # (1,)
                    + stats_counts  # (15,)
                    + [top_discard]  # (1,)
                    + [self.hand_card]  # (1,)
                    + player_obs  # (12,) or (num_players * 12,)
                ),
            )

        elif self.observation_mode == "simple_port_to_other":

            player_obs = [5 if x == 15 else x for x in player_obs]

            obs = np.array(
                (
                    [min(max(self.global_turn_counter, 0), 127)] # (1,)
                    + [self.expected_action[1] == "place"] # (1,)
                    + [min(cards_sum.min(), 127)]  # (1,)
                    + [n_hidden.min()]  # (1,)
                    + stats_counts  # (15,)
                    + [top_discard]  # (1,)
                    + [self.hand_card]  # (1,)
                    + player_obs  # (12,) or (num_players * 12,)
                ),
            )

        elif self.observation_mode == "efficient_one_hot_port_to_other":
            for ind, val in enumerate(player_obs):
                if val == self.fill_masked_unknown_value:
                    efficient_one_hot_obs[ind] = 1

            player_obs = [5 if x == self.fill_masked_unknown_value else x for x in player_obs]

            extended_player_obs = []
            if self.observe_other_player_indirect:
                extended_player_obs.append(player_obs)
                extended_player_obs.append(efficient_one_hot_obs)
            else:
                for player in range(self.num_players):
                    extended_player_obs.append(player_obs[player*12:(player + 1)*12])
                    extended_player_obs.append(efficient_one_hot_obs[player*12:(player + 1)*12])

            extended_player_obs = np.concatenate(extended_player_obs, axis=0)

            obs = np.concatenate([[min(max(self.global_turn_counter, 0), 127)], [self.expected_action[1] == "place"], global_stats, [top_discard], [self.hand_card], extended_player_obs])
            #     sizes:      17               , 1           ,    1        ,     24 or 24*num_players            

        elif self.observation_mode == "onehot":
            # one-hot encode top_discard & hand_card => shape (17,) each
            top_discard_oh = self._one_hot_card_value(top_discard)
            hand_card_oh = self._one_hot_card_value(self.hand_card)

            # one-hot encode player_obs => shape (Ncards * 17,)
            #   Ncards = 12 if indirect else num_players*12 if direct
            player_obs_oh = self._one_hot_encode_array(player_obs)

            # final observation -> concat everything
            obs = np.concatenate([global_stats, top_discard_oh, hand_card_oh, player_obs_oh])

        elif self.observation_mode == "efficient_one_hot":
            for ind, val in enumerate(player_obs):
                if val == self.fill_masked_unknown_value:
                    efficient_one_hot_obs[ind] = 1

            player_obs = [5 if x == self.fill_masked_unknown_value else x for x in player_obs]

            extended_player_obs = []
            if self.observe_other_player_indirect:
                extended_player_obs.append(player_obs)
                extended_player_obs.append(efficient_one_hot_obs)
            else:
                for player in range(self.num_players):
                    extended_player_obs.append(player_obs[player*12:(player + 1)*12])
                    extended_player_obs.append(efficient_one_hot_obs[player*12:(player + 1)*12])

            extended_player_obs = np.concatenate(extended_player_obs, axis=0)

            obs = np.concatenate([global_stats, [top_discard], [self.hand_card], extended_player_obs])
            #     sizes:      17               , 1           ,    1        ,     24 or 24*num_players
        # verify shape
        assert obs.shape == self.obs_shape, (
            f"Unexpected observation shape {obs.shape}, expected {self.obs_shape}"
        )

        action_mask = self._jit_action_mask(
            self.players_masked, player_id, self.expected_action[1], self.previous_action
        )
        return obs, action_mask

    def collect_hidden_card_sums(self) -> Tuple[np.array, np.array]:
        """
        Returns array of number of hidden cards of each player (np.array[]) +
            the card sums of each player (np.array[])
        """
        # get global stats
        (
            stats_counts,
            cards_sum,
            n_hidden,
            top_discard,
        ) = self._jit_observe_global_game_stats(
            self.players_cards,
            self.players_masked,
            np.array(self.discard_pile, dtype=self.players_cards.dtype),
            count_players_cards=not self.observe_other_player_indirect,
        )
        return n_hidden, cards_sum

    @staticmethod
    @njit(fastmath=True)
    def _jit_action_mask(
        players_masked: np.ndarray,
        player_id: int,
        next_action: str,
        previous_action: int,
        action_mask_shape=(26,),
    ):
        if next_action == "place":
            # must be either a card that is front of the player (0 is for refunded)
            mask_place = (players_masked[player_id] != 0).astype(np.int8) #0-12 is place
            # discard hand card and reveal an masked card
            if previous_action == 24:  # 24 is draw from drawpile
                mask_place2 = (players_masked[player_id] == 2).astype(np.int8)
            else:  # 25 is draw from discard pile
                mask_place2 = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_draw = np.zeros(2, dtype=np.int8)
        else:  # draw
            # only draw allowed
            mask_place = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_place2 = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_draw = np.ones(2, dtype=np.int8)
        action_mask = np.concatenate((mask_place, mask_place2, mask_draw))
        assert (
            action_mask.shape == action_mask_shape
        ), "action mask needs to have shape (26,)"
        return action_mask

    @staticmethod
    @njit(fastmath=True)
    def _jit_observe_global_game_stats(
        players_cards: np.ndarray,
        players_masked: List[int],
        pile: np.ndarray,
        count_players_cards: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """observe game statistics, features to percieve global game"""
        # pile plus placeholders for all possible card values
        counted = np.array(list(pile) + list(range(-2, 13)), dtype=players_cards.dtype)
        known_cards_sum = [0] * players_cards.shape[0]
        count_hidden = [0] * players_cards.shape[0]

        masked_option = players_masked == 1
        for pl in range(players_cards.shape[0]):
            cards_pl = players_cards[pl][masked_option[pl]]
            if count_players_cards:
                counted = np.concatenate((counted, cards_pl))
            # player sums
            known_cards_sum[pl] = cards_pl.sum()

        counts = np.bincount(counted - np.min(counted)) - 1
        # not unknown
        masked_option_hidden = players_masked == 2
        for pl in range(masked_option_hidden.shape[0]):
            count_hidden[pl] = np.sum(masked_option_hidden[pl])

        pile_top = pile[-1] if len(pile) else -3  # fallback if needed
        known_cards_sum = np.array(known_cards_sum)
        count_hidden = np.array(count_hidden)
        return list(counts.flatten()), known_cards_sum, count_hidden, pile_top

    @staticmethod
    @njit(fastmath=True)
    def _jit_known_player_cards(
        players_cards,
        players_masked,
        player_id: int,
        fill_unknown=np.nan,
    ) -> np.array:
        """
        get array of player's own 12 cards; mask unknown with fill_unknown
        mask refunded with fill_unknown as well (the environment can interpret them differently)
        """
        cards = np.full_like(players_cards[player_id], fill_unknown)
        masked_revealed = players_masked[player_id] != 2
        cards[masked_revealed] = players_cards[player_id][masked_revealed]
        return list(cards.flatten())

    @staticmethod
    @njit(fastmath=True)
    def _jit_known_player_cards_all(
        players_cards,
        players_masked,
        player_id: int,
        fill_unknown=np.nan,
    ) -> np.array:
        """
        get array of cards for all players, with unknown/refunded masked with fill_unknown,
        in the order: current_player first, then the rest
        """
        cards = np.full_like(players_cards, fill_unknown)
        all_ids = np.roll(np.arange(players_cards.shape[0]), - player_id)
        for i,pl in enumerate(all_ids):
            masked_revealed = players_masked[pl] != 2
            cards[i][masked_revealed] = players_cards[pl][masked_revealed]
        
        return list(cards.flatten())

    # [end: collect observation]

    # [start: perform actions]
    def act(self, player_id: int, action_int: int):
        """perform actions

        returns:
            game_over: If this was the last action and the game is now over
            last_action: If this was the last action for the player
        """
        assert self.expected_action[0] == player_id, (
            f"ILLEGAL ACTION: expected {self.expected_action[0]}"
            f" but requested was {player_id}"
        )
        assert action_int is not None, "ILLEGAL ACTION: None not supported"
        assert 0 <= action_int <= 25, f"action int {action_int} not in range(0,26)"

        if self.has_terminated:
            warnings.warn(
                "Attempt playing a terminated game."
                " The game has already ended."
            )
            return True, True

        game_over = False
        last_action = False
        if 24 <= action_int <= 25:
            # draw
            assert self.hand_card == self.fill_masked_unknown_value, (
                "ILLEGAL ACTION. requested draw action but hand_card not empty."
            )
            self._action_draw_card(player_id, action_int)
        else:
            # place
            assert self.hand_card != self.fill_masked_unknown_value, (
                "ILLEGAL ACTION. requested place action but no card in hand."
            )
            self._action_place(player_id, action_int)

            # Check if the player has revealed all cards
            last_action = self._player_goal_check(self.players_masked, player_id)
            if last_action and self.last_round_initiator is None:
                self.last_round_initiator = player_id

            if self.last_round_initiator is not None:
                last_action = True

        if player_id == self.num_players - 1 and self.expected_action[1] == self._name_draw and self.global_turn_counter<127:
            self.global_turn_counter += 1

        self.previous_action = action_int
        self._internal_next_action()

        # Check if the next action belongs to the player that initiated the last round
        if self.expected_action[0] == self.last_round_initiator:
            self.has_terminated = True
            self.game_metrics["final_score"] = self._evaluate_game(
                self.players_cards, player_id, score_penalty=self.score_penalty
            )
            game_over = True

        return game_over, last_action

    def _action_draw_card(self, player_id: int, draw_from: int):
        """
        args:
            player_id: int, player who is playing
            from_drawpile: bool, action: True to draw from drawpile, else discard pile

        returns:
            game over: bool winner_id
            final_scores: list(len(n_players)) if game over
        """
        
        # drawing action
        if draw_from == 24:
            # draw from drawpile
            if not self.drawpile:
                # cardpile is empty, reshuffle.
                self.drawpile, self.discard_pile = self._reshuffle_discard_pile(
                    np.array(self.discard_pile, dtype=self.card_dtype)
                )
            self.hand_card = self.drawpile.pop()
        else:
            # draw from discard
            self.hand_card = self.discard_pile.pop()

        # action done

    def _action_place(
        self, player_id: int, action_place_to_pos: int
    ) -> Tuple[bool, np.ndarray]:
        """
        args:
            player_id: int, player who is playing
            action_place_to_pos: int, action between 0 and 11,

        returns:
            game over: bool winner_id
            final_scores: list(len(n_players)) if game over
        """

        if action_place_to_pos in range(0, 12):
            # replace one of the 0-11 cards with hand card of player
            # unmask new card
            # discard deck card
            self.discard_pile.append(self.players_cards[player_id][action_place_to_pos])
            self.players_masked[player_id][action_place_to_pos] = 1
            self.players_cards[player_id][action_place_to_pos] = self.hand_card
        else:
            # 12..23 -> discard the hand card, reveal an unrevealed card
            assert self.previous_action == 24, (
                f"ILLEGAL ACTION. Can't place the card back on the discard pile "
                f"unless it was drawn from the drawpile."
            )
            place_pos = action_place_to_pos - 12
            assert self.players_masked[player_id][place_pos] == 2, (
                f"illegal action: card {place_pos} is already revealed for player {player_id}"
            )
            self.discard_pile.append(self.hand_card)
            self.players_masked[player_id][place_pos] = 1

        # check if three in a row => refunded
        (
            is_updated,
            pc_update,
            pm_update,
            dp_add,
        ) = self._remask_refunded_player_cards_jit(
            self.players_cards,
            self.players_masked,
            player_id,
            self.fill_masked_refunded_value,
        )
        if is_updated:
            self.game_metrics["num_refunded"][player_id] += 1
            self.players_cards, self.players_masked = pc_update, pm_update
            self.discard_pile.extend(dp_add)

        self.game_metrics["num_placed"][player_id] += 1
        self.hand_card = self.fill_masked_unknown_value

    @staticmethod
    @njit(fastmath=True)
    def _remask_refunded_player_cards_jit(
        players_cards,
        players_masked,
        player_id: int,
        fill_masked_refunded_value: int = -14,
    ):
        """
        check if any stack of 3 cards is the same => refunded
        replace them with fill_masked_refunded_value and players_masked=0
        """
        cards_to_discard_pile = np.empty((0,), dtype=np.int8)
        values_updated = False

        # each stack is 3 cards (4 columns => indices 0..11 in sets of 3)
        for stack in range(players_cards[player_id].shape[0] // 3):
            slice_tup = slice(stack * 3, stack * 3 + 3, 1)
            cards_stack_3_tup = players_cards[player_id][slice_tup]
            # check if all are the same
            if np.min(cards_stack_3_tup) == np.max(cards_stack_3_tup):
                # also check if they are actually revealed
                if np.all(players_masked[player_id][slice_tup] == 1):
                    players_masked[player_id][slice_tup] = 0
                    # note: we can store the masked values into discard pile if needed
                    cards_to_discard_pile = np.append(
                        cards_to_discard_pile, players_masked[player_id][slice_tup]
                    )
                    players_cards[player_id][slice_tup] = fill_masked_refunded_value
                    values_updated = True
        if values_updated:
            return (
                values_updated,
                players_cards,
                players_masked,
                list(cards_to_discard_pile),
            )
        else:
            return values_updated, None, None, None

    @staticmethod
    @njit(fastmath=True)
    def _player_goal_check(players_masked, player_id):
        """check if game over, when player_id has all cards known (=!2)"""
        return np.all((players_masked[player_id] != 2))

    @staticmethod
    @njit(fastmath=True)
    def _evaluate_game(
        players_cards, player_won_id, score_penalty: float = 2.0
    ) -> List[int]:
        """
        calculate game scores
        """
        score = [0.0] * players_cards.shape[0]

        for pl in range(players_cards.shape[0]):
            for stack in range(players_cards[pl].shape[0] // 3):
                slice_tup = slice(stack * 3, stack * 3 + 3, 1)
                cards_stack_3_tup = players_cards[pl][slice_tup]
                # only sum up if not refunded
                if np.min(cards_stack_3_tup) != np.max(cards_stack_3_tup):
                    score[pl] += np.sum(cards_stack_3_tup)

        # penalty if finisher is not the actual winner
        if min(score) != score[player_won_id]:
            score[player_won_id] *= score_penalty
        return score

    def get_game_metrics(self):
        return self.game_metrics

    def get_expected_action(self):
        return self.expected_action.copy()

    # [start: render utils]

    def render_table(self):
        """
        render game:
            - render cards for all players
            - render game statistics
        """
        render_cards_open = False
        str_board = f"{'='*7} render board: {'='*5} \n"
        str_board += self._render_game_stats()
        if self.has_terminated:
            res = dict(
                zip(list(range(self.num_players)), self.game_metrics["final_score"])
            )
            str_board += f"{'='*7} GAME DONE {'='*8} \n" f"Results: {res} \n"
            render_cards_open = True
        for pl in range(self.num_players):
            str_board += self.render_player(pl, render_cards_open)
        return str_board

    def _render_game_stats(self):
        """render game statistics"""
        card_hand = self.hand_card if -2 <= self.hand_card <= 12 else "empty"
        discard_pile_top = self.discard_pile[-1] if self.discard_pile else "empty"
        str_stats = (
            f"{'='*7} stats {'='*12} \n"
            f"next turn: {self.expected_action[1]} "
            f"by Player {self.expected_action[0]} \n"
            f"holding card player {self.expected_action[0]}: "
            f"{card_hand} \n"
            f"discard pile top: {discard_pile_top} \n"
        )

        return str_stats

    def _render_player_cards(self, player_id, render_cards_open):
        array = self.players_cards[player_id].astype(np.str_)

        if render_cards_open:
            array[self.players_masked[player_id] == 2] = np.char.add(
                np.array(["u"], dtype=np.str_),
                array[self.players_masked[player_id] == 2],
            )
        else:
            array[self.players_masked[player_id] == 2] = "u"

        array[self.players_masked[player_id] == 0] = "d"
        array = array.reshape(4, -1).T
        array = np.array2string(
            array, separator="\t ", formatter={"str_kind": lambda x: str(x)}
        )
        return array

    def render_player(self, player_id, render_cards_open=False):
        """render cards of 1 player"""
        str_pl = f"{'='*7} Player {player_id} {'='*10} \n"
        str_pl += self._render_player_cards(player_id, render_cards_open) + "\n"
        return str_pl

    @classmethod
    def render_action_explainer(cls, action_int: int):
        """adds a string explaining actions to plot of render_player"""
        assert action_int in range(0, 26), f"action not valid: {action_int}"

        if action_int == 24:
            return "draw from drawpile"
        elif action_int == 25:
            return "draw from discard pile"

        if action_int in range(0, 12):
            place_id = action_int
            result = f"place card ({action_int}) - "
        else:  # 12..23
            place_id = action_int - 12
            result = f"handcard discard & reveal card ({action_int}) - "

        col = math.floor(place_id / 3)
        row = place_id % 4
        result += f"col:{col} row:{row}"
        return result

    @classmethod
    def render_actions(cls):
        """possible actions"""
        array = np.char.add(np.arange(12).reshape(4, -1).T.astype(np.str_), "/")
        array = np.char.add(array, np.arange(12, 24).reshape(4, -1).T.astype(np.str_))
        array = np.array2string(
            array, separator="\t ", formatter={"str_kind": lambda x: str(x)}
        )
        return (
            f"action ids 0-25:\n"
            f" (put handcard here / reveal this card)\n {array}\n"
            f"24: draw from drawpile\n"
            f"25: draw from discard pile"
        )
    # [end: render utils]


if __name__ == "__main__":
    # simple test
    game = SkyjoGame()
    obs, mask = game.collect_observation(player_id=0)
    print("Obs shape:", obs.shape)
    print("Obs:", obs)
    print("Action mask:", mask)