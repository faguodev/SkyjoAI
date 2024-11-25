from typing import List, Dict

import numpy as np
from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from environment.skyjo_game import SkyjoGame

DEFAULT_CONFIG = {
    "num_players": 3,
    "score_penalty": 2.0,
    "observe_other_player_indirect": True,
    "mean_reward": 1.0,
    "reward_refunded": 0.001,
    "render_mode": "human"
}


def env(**kwargs):
    """wrap SkyJoEnv in"""
    env = SimpleSkyjoEnv(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class SimpleSkyjoEnv(AECEnv):

    metadata = {
        "render_modes": ["human"],
        "name": "skyjo",
        "is_parallelizable": False,
        "video.frames_per_second": 1,
    }

    def __init__(
        self,
        num_players=2,
        score_penalty: float = 2.0,
        observe_other_player_indirect: bool = False,
        mean_reward: float = 1.0,
        reward_refunded: float = 0.0,
        render_mode = None
    ):
        """
        Pettingzoo Gym for the card game SkyJo

        params:
            # game configuration
            num_players: int, number of players
            score_penalty: float, game default is 2.0
                score penalty for players ending but not winning the game

            # observation space configuration
            observe_other_player_indirect: bool
                True: observation space is:
                    game statistics (pile +  player cards):
                    + own 12 player cards
                False: observation space is:
                    game statistics (excluding player cards)
                    + player cards of every player

            # rewards
            mean_reward: float, default: 1.0
                mean reward at the end of an game
                recommended to be > 0, e.g. Environments (like RLLib)
                are positive sum games
            reward_refunded: float, default: 0.0
                adds an additional reward to learn the concept
                of refunding cards in skyjo

        observation space is DictSpace:
            observations:
                (1,) lowest sum of players, calculated feature
                (1,) lowest number of unmasked cards of any player,
                    calculated feature
                (15,) counts of cards past discard pile cards & open player cards,
                    calculated feature
                (1,) top discard pile card
                (1,) current hand_card
                total: (19,)

                if observe_other_player_indirect is True:
                    # constant for any num_players
                    (12) own cards
                    total: (31,)
                elif observe_other_player_indirect is False:
                    (num_players*4*3,)
                    total: (19+12*num_players,)

            action_mask:
                (26,)

        action_space is Discrete(26):
            0-11: place hand card to position 0-11
            12-23: discard place hand card and reveal position 0-11
            24: pick hand card from drawpile
            25: pick hand card from discard pile

        """
        super().__init__()

        # Hyperparams
        self.num_players = num_players
        self.mean_reward = mean_reward
        self.reward_refunded = reward_refunded

        self.table = SkyjoGame(
            num_players,
            score_penalty=score_penalty,
            observe_other_player_indirect=observe_other_player_indirect,
        )

        #rewards stuff
        self.final_reward, self.score_per_unknown = self.read_reward_params("reward_parameter.txt")

        # start PettingZoo API stuff
        self.render_mode = render_mode
        self.agents = [i for i in range(num_players)]
        self.possible_agents = self.agents[:]

        self.agent_selection = self._expected_agentname_and_action()[0]

        self.terminations = self._convert_to_dict([False for _ in range(self.num_agents)])
        # Not needed, since SkyJo will always end, but required by AECEnv
        self.truncations = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = {i: {} for i in self.agents}

        self._observation_spaces = spaces.Dict(
            {
                agent_id: spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-24,
                            high=127,
                            shape=self.table.obs_shape,
                            dtype=self.table.card_dtype,
                        ),
                        "action_mask": spaces.Box(
                            low=0,
                            high=1,
                            shape=self.table.action_mask_shape,
                            dtype=np.int8,
                        ),
                    }
                )
                for agent_id in self.possible_agents
            }
        )
        self._action_spaces = spaces.Dict(
            {
                agent_id: spaces.Discrete(self.table.action_mask_shape[0])
                for agent_id in self.possible_agents
            }
        )
        # end obs / actions space
        # end PettingZoo API stuff

    def observation_space(self, agent):
        """
        observations are:
            (1,) lowest sum of players, calculated feature
            (1,) lowest number of unmasked cards of any player,
                calculated feature
            (15,) counts of cards past discard pile cards & open player cards,
                calculated feature
            (1,) top discard pile card
            (1,) current hand_card
            total: (19,)

            if observe_other_player_indirect is True:
                # constant for any num_players
                (12) own cards
                total: (31,)
            elif observe_other_player_indirect is False:
                (num_players*4*3,)
                total: (19+12*num_players,)

        Args:
            agent ([type]): agent string

        Returns:
            gym.space: observation_space of agent
        """
        return self._observation_spaces[agent]

    def action_space(self, agent):
        """part of the PettingZoo API
        action_space is Discrete(26):
            0-11: place hand card to position 0-11
            12-23: discard place hand card and reveal position 0-11
            24: pick hand card from drawpile
            25: pick hand card from discard pile

        Args:
            agent ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return self._action_spaces[agent]

    def observe(self, agent: str) -> Dict[str,np.ndarray]:
        """
        get observation and action mask from environment
        part of the PettingZoo API]

        Args:
            agent ([str]): agent string

        Returns:
            dict: {"observations": np.ndarray, "action_mask": np.ndarray}
        """
        
        obs, action_mask = self.table.collect_observation(
            agent #self._name_to_player_id(agent)
        )
        return {"observations": obs, "action_mask": action_mask}

    def step(self, action: int) -> None:
        """part of the PettingZoo API

        Args:
            action (int): 
                action is number from 0-25:
                    0-11: place hand card to position 0-11
                    12-23: discard place hand card and reveal position 0-11
                    24: pick hand card from drawpile
                    25: pick hand card from discard pile
            
        Returns:
            None: 
        """  
        current_agent = self.agent_selection
        #Calculate Score before action
        n_hidden_cards, Card_sum= self.table.collect_hidden_card_sums() #both should be arrays of length: num players
        assert(len(n_hidden_cards) == len(Card_sum) and len(n_hidden_cards) == self.num_players)
        score_before = Card_sum[current_agent] + self.score_per_unknown * n_hidden_cards[current_agent]

        # if was done before
        if self.terminations[current_agent]:
            return self._was_dead_step(None)

        game_over, last_action = self.table.act(current_agent, action_int=action)

        #Calc score after action: first gather obs
        n_hidden_cards, Card_sum = self.table.collect_hidden_card_sums()
        score_after = Card_sum[current_agent] + self.score_per_unkown * n_hidden_cards[current_agent]
        self.rewards = score_before - score_after
        # action done, rewards if game over
        if game_over:
            # current player has terminated the game for all. gather rewards
            self.rewards = self._convert_to_dict(
                self._calc_final_rewards(**(self.table.get_game_metrics()))
            )
            self.terminations = {i: True for i in self.agents}

        if last_action:
            self.truncations[current_agent] = True

        # done
        self._accumulate_rewards()

        # prepare for next agent
        if not game_over:
            self.agent_selection = self._expected_agentname_and_action()[0]

    def reset(self, seed: int = None, options=None) -> None:
        """
        reset the environment
        part of the PettingZoo API
        """
        self.table.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._expected_agentname_and_action()[0]
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)]
        )
        self.terminations = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.truncations = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = {i: {} for i in self.agents}

        if seed is not None:
            self.table.set_seed(seed)

    def render(self) -> None:
        """render board of the game to stdout

        part of the PettingZoo API"""
        if self.render_mode == "human":
            print(self.table.render_table())


    def close(self) -> None:
        """part of the PettingZoo API"""
        pass

    # start utils

    def _calc_final_rewards(
            self, final_score: List[int], **kwargs
    ):
        """
        get reward from score.
        reward is 100 for winner and -100 for all loosers

        args:
            game_results: dict['str': np.array of len(players) e.g. np.array([35,65,50])

        returns:
            reward: np.array [len(players)] e.g. np.array([ 16,-14,+1])
        """

        final_scores = np.asarray(final_score)
        winner = np.where(final_scores == np.min(final_scores))

        rewards = np.asarray([self.final_reward if winner == i else -self.final_reward for i in
                              range(len(final_scores))])  # should create an array of rewards
        # either set to 100 if i==winner else to -100
        return rewards

    def _calc_final_rewards_old(
        self, final_score: List[int], num_refunded: List[int], **kwargs
    ):
        """
        get reward from score.
        reward is relative performance to average score
        default mean reward is self.mean_reward == 1

        args:
            game_results: dict['str': np.array of len(players) e.g. np.array([35,65,50])

        returns:
            reward: np.array [len(players)] e.g. np.array([ 16,-14,+1])
        """
        score = np.array(final_score)
        reward = -score + np.mean(score) + self.mean_reward

        if self.reward_refunded:
            reward += np.array(num_refunded) * self.reward_refunded
        return reward

    @staticmethod
    def _name_to_player_id(name: str) -> int:
        """[convert agent name to int  e.g. player_1 to int(1)]

        Args:
            name (str): agent name

        Returns:
            int: agent int 
        """        
        return int(name.split("_")[-1])

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _expected_agentname_and_action(self):
        """implemented, get next player name for action from skyjo"""
        a = self.table.get_expected_action()
        return a[0], a[1]

    def read_reward_params(file_name):

        file = open(file_name, "r")
        lines = file.readlines()

        final_reward_value = float(lines[0].split(":")[1].strip())
        score_per_unknown_card = float(lines[1].split(":")[1].strip())

        return final_reward_value, score_per_unknown_card

    # end utils


if __name__ == "__main__":
    from pettingzoo.test import api_test
    env = SimpleSkyjoEnv()
    api_test(env, num_cycles=1000, verbose_progress=True)