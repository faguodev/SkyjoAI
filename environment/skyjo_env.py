from typing import List, Dict
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from environment.skyjo_game import SkyjoGame
from typing import TypedDict

class RewardConfig(TypedDict, total=False):
    score_penalty: float
    reward_refunded: float
    final_reward: float
    score_per_unknown: float
    action_reward_reduction: float
    final_reward_offset: float
    curiosity_reward: float
    old_reward: bool

def default_reward_config() -> RewardConfig:
    return {
        "score_penalty": 2.0, # Seems useless
        "reward_refunded": 0.0,
        "final_reward": 100.0,
        "score_per_unknown": 5.0,
        "action_reward_reduction": 1.0,
        "final_reward_offset": 0.0,
        "curiosity_reward": 4.0,
        "old_reward": False,
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
        observe_other_player_indirect: bool = False,
        observation_mode: str = "simple",
        reward_config: RewardConfig = None,
        render_mode=None
    ):
        """
        PettingZoo AEC Env for SkyJo
        """
        super().__init__()

        self.num_players = num_players
        self.observation_mode = observation_mode
        self.reward_config: RewardConfig = {**default_reward_config(), **(reward_config or {})}


        self.table = SkyjoGame(
            num_players=num_players,
            score_penalty=self.reward_config["score_penalty"],
            observe_other_player_indirect=observe_other_player_indirect,
            observation_mode=self.observation_mode,
        )

        self.render_mode = render_mode
        self.agents = [i for i in range(num_players)]
        self.possible_agents = self.agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.agent_selection = self._expected_agentname_and_action()[0]

        # Use reward parameters from the config
        self.reward_refunded = self.reward_config["reward_refunded"]
        self.final_reward = self.reward_config["final_reward"]
        self.action_reward_reduction = self.reward_config["action_reward_reduction"]
        self.final_reward_offset = self.reward_config["final_reward_offset"]
        self.score_per_unknown = self.reward_config["score_per_unknown"]
        self.old_reward = self.reward_config["old_reward"]
        self.curiosity_reward = self.reward_config["curiosity_reward"]

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

    def update_action_reward_reduction(self, action_reward_reduction):
        self.action_reward_reduction = action_reward_reduction

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

        # if was done before
        if self.terminations[current_agent]:
            return self._was_dead_step(None)

        curious = False
        if action >= 12 and action < 24:
            curious = True

        # reward shaping: score before
        n_hidden, card_sum = self.table.collect_hidden_card_sums()
        score_before = card_sum[current_agent] + self.score_per_unknown * n_hidden[current_agent]

        game_over, last_action = self.table.act(current_agent, action_int=action)

        # score after
        n_hidden, card_sum = self.table.collect_hidden_card_sums()
        score_after = card_sum[current_agent] + self.score_per_unknown * n_hidden[current_agent]

        if not game_over:
            # player revealed all => we give final reward offset
        #     self.rewards[current_agent] = self.final_reward_offset - card_sum[current_agent]
        # else:
            # simple delta-based reward
            if score_before - score_after <= 0:
                self.infos[current_agent]["undesirable_action"] += 1
            self.rewards[current_agent] = self.action_reward_reduction * (score_before - score_after)
            if curious:
                self.rewards[current_agent] += self.curiosity_reward

        # if the game is over, finalize
        else:
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
            
                # TODO: If we first only pretrain and then only self-play then this might not be necessary.
                # Otherwise, we still might need to adapt this due to floating point values introduced by
                # reward decay. Perhaps rounding other rewards can be a solution.

            final_scores = [int(score) for score in self.table.get_game_metrics()["final_score"]]
            winner_ids = np.argwhere(final_scores == np.min(final_scores)).flatten().tolist()
            self.infos[0]["winner_ids"] = winner_ids
            self.infos[0]["final_scores"] = final_scores

        if last_action:
            self.truncations[current_agent] = True

        self._accumulate_rewards()

        # next agent
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
        self.rewards = self._convert_to_dict([0] * self.num_agents)
        self._cumulative_rewards = self._convert_to_dict([0] * self.num_agents)
        self.terminations = self._convert_to_dict([False] * self.num_agents)
        self.truncations = self._convert_to_dict([False] * self.num_agents)
        self.infos = {i: {"undesirable_action": 0} for i in self.agents}

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