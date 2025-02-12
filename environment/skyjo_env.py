from typing import List, Dict
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

# IMPORTANT: import the updated SkyjoGame from your local file
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
    """wrap SkyJoEnv in PettingZoo wrappers"""
    _env = SimpleSkyjoEnv(**kwargs)
    _env = wrappers.CaptureStdoutWrapper(_env)
    _env = wrappers.TerminateIllegalWrapper(_env, illegal_reward=-1)
    _env = wrappers.AssertOutOfBoundsWrapper(_env)
    _env = wrappers.OrderEnforcingWrapper(_env)
    return _env


class SimpleSkyjoEnv(AECEnv):
    metadata = {
        "render_modes": ["human", None],
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
        final_reward: float = 100.0,
        score_per_unknown: float = 5.0,
        action_reward_decay: float = 1.0,
        final_reward_offest: float = 0.0,
        old_reward: bool = False,
        render_mode=None,
    ):
        """
        PettingZoo AEC Env for SkyJo
        """
        super().__init__()

        self.num_players = num_players
        self.mean_reward = mean_reward
        self.reward_refunded = reward_refunded
        self.final_reward = final_reward
        self.action_reward_decay = action_reward_decay
        self.final_reward_offest = final_reward_offest
        self.score_per_unknown = score_per_unknown
        self.old_reward = old_reward
        
        self.table = SkyjoGame(
            num_players=num_players,
            score_penalty=score_penalty,
            observe_other_player_indirect=observe_other_player_indirect,
        )

        self.render_mode = render_mode
        self.agents = [i for i in range(num_players)]
        self.possible_agents = self.agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.agent_selection = self._expected_agentname_and_action()[0]

        # termination/truncation
        self.terminations = self._convert_to_dict([False] * self.num_agents)
        self.truncations = self._convert_to_dict([False] * self.num_agents)
        self.infos = {i: {} for i in self.agents}

        # define observation & action space
        self._observation_spaces = self._convert_to_dict(
            [
                spaces.Dict(
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
                for _ in self.possible_agents
            ]
        )
        self._action_spaces = self._convert_to_dict(
            [
                spaces.Discrete(self.table.action_mask_shape[0])
                for _ in self.possible_agents
            ]
        )
        # end obs / actions space
        # end PettingZoo API stuff

    def update_action_reward_decay(self, action_reward_decay):
        self.action_reward_decay = action_reward_decay
        #print(action_reward_decay)

    def observation_space(self, agent):
        """
        observations are:
            (1,) lowest sum of players, calculated feature
            (1,) lowest number of unmasked cards of any player,
                calculated feature
            (15,) counts of cards past discard pile cards & open player cards,
                calculated feature
            (17) top discard pile card
            (17,) current hand_card
            total: (51,)

            if observe_other_player_indirect is True:
                # constant for any num_players
                (12*17) own cards
                total: (51+12*17,)
            elif observe_other_player_indirect is False:
                (num_players*4*3*17,)
                total: (51+12*num_players*17,)

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
        #print(current_agent)

        # if was done before
        if self.terminations[current_agent]:
            return self._was_dead_step(None)

        # reward shaping: score before
        _, card_sum = self.table.collect_hidden_card_sums()
        score_before = card_sum[current_agent]

        game_over, last_action = self.table.act(current_agent, action_int=action)

        hidden_c = self.observe(current_agent)["observations"][15::17]
        #print(f"In Env_class - hidden cards of actor {current_agent}",[i for i, val in enumerate(hidden_c[3:15]) if val == 1])

        # score after
        n_hidden_cards, card_sum = self.table.collect_hidden_card_sums()
        score_after = card_sum[current_agent]

        if not self.old_reward:
            if last_action:
                # player revealed all => we give final reward offset
                self.rewards[current_agent] = self.final_reward_offest - 3 * card_sum[current_agent]
            else:
                # simple delta-based reward
                self.rewards[current_agent] = self.action_reward_decay * (score_before - score_after)

        # if the game is over, finalize
        if game_over:
            #print("Reward_decay = ", self.action_reward_decay)
            #if self.old_reward:
            self.rewards = self._convert_to_dict(
                self._calc_final_rewards(**(self.table.get_game_metrics()))
            )
            self.terminations = {i: True for i in self.agents}


            # --- NEW PART: store final card sums in info dict for each agent ---
            n_hidden, card_sums = self.table.collect_hidden_card_sums()
            for idx, agent_id in enumerate(self.agents):
                self.infos[agent_id]["final_card_sum"] = card_sums[idx]
                self.infos[agent_id]["n_hidden_cards"] = n_hidden[idx]
            
                temp = max((self.rewards).values())
                winners = [key for key in self.rewards if self.rewards[key] == temp]
                if not self.old_reward:
                    for w in winners:
                        self.rewards[w] += 0.5
                self.infos[agent_id]["winner_ids"] = winners

        if last_action:
            self.truncations[current_agent] = True

        self._accumulate_rewards()

        # next agent
        if not game_over:
            self.agent_selection = self._expected_agentname_and_action()[0]
        
        #if self.agent_selection != current_agent

    def reset(self, seed: int = None, options=None) -> None:
        """
        reset the environment
        part of the PettingZoo API
        """
        self.table.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._expected_agentname_and_action()[0]
        self.rewards = self._convert_to_dict([0] * self.num_agents)
        self._cumulative_rewards = self._convert_to_dict([0] * self.num_agents)
        self.terminations = self._convert_to_dict([False] * self.num_agents)
        self.truncations = self._convert_to_dict([False] * self.num_agents)
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


    # --------------------------- UTILITIES --------------------------- #

    def _calc_final_rewards(
            self, final_score: List[int], **kwargs
    ):
        """
        reward is +100 for winner, -100 for others
        """
        final_scores = np.asarray(final_score)
        winner_ids = np.where(final_scores == np.min(final_scores))[0]
        # for simplicity, pick the first if tie
        main_winner = winner_ids[0]
        rewards = []
        for i in range(len(final_scores)):
            if i == main_winner:
                rewards.append(self.final_reward)
            else:
                rewards.append(-self.final_reward)
        return rewards

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
        """Retrieve which player_id + action name is expected next by the game."""
        a = self.table.get_expected_action()  # e.g. [player_id, 'draw' or 'place']
        return a[0], a[1]

    def read_reward_params(self, file_name):
        with open(file_name, "r") as file:
            lines = file.readlines()
        final_reward_value = float(lines[0].split(":")[1].strip())
        score_per_unknown_card = float(lines[1].split(":")[1].strip())
        return final_reward_value, score_per_unknown_card


if __name__ == "__main__":
    from pettingzoo.test import api_test
    env = SimpleSkyjoEnv()
    api_test(env, num_cycles=1000, verbose_progress=True)
