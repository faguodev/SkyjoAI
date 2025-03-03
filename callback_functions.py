import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics import (ENV_RUNNER_RESULTS)

class RewardDecay_Callback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Decay the reward scaling factor over training iterations
        action_reward_reduction = max(0.05, 1.0 - result["training_iteration"] * 0.005)
        algorithm.config.env_config["action_reward_reduction"] = action_reward_reduction

class SkyjoLogging_and_SelfPlayCallbacks(DefaultCallbacks):

    def __init__(self, main_policy_id, win_rate_threshold):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
        self.winning_policy = []
        self.win_rate_threshold = win_rate_threshold
        self.main_policy_id = main_policy_id

    def on_episode_end(        
        self,
        *,
        episode,
        **kwargs,): #policies, 
        """
        This is called at the end of each episode. We grab
        the final card sum from the `info` dict for each agent
        and log it as a custom metric.
        """
        
        #win_id = 0
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info is None:
                continue
            if "final_sum_of_revealed_cards" in info:
                metric_name = f"final_sum_of_revealed_cards_{agent_id}"
                if metric_name not in episode.hist_data:
                    episode.hist_data[metric_name] = []
                episode.hist_data[metric_name].append(info["final_sum_of_revealed_cards"])
            if "n_hidden_cards" in info:
                metric_name = f"n_hidden_cards_{agent_id}"
                if metric_name not in episode.hist_data:
                    episode.hist_data[metric_name] = []
                episode.hist_data[metric_name].append(info["n_hidden_cards"])
            if "undesirable_action" in info:
                metric_name = f"undesirable_action_{agent_id}"
                if metric_name not in episode.hist_data:
                    episode.hist_data[metric_name] = []
                episode.hist_data[metric_name].append(info["undesirable_action"])
            # The is only stored in the first agent's infos dict
            if "winner_ids" in info:
                metric_name = "winner_ids"
                if metric_name not in episode.hist_data:
                    episode.hist_data[metric_name] = []
                episode.hist_data[metric_name].append(info[metric_name])
            if "final_score" in info:
                metric_name = f"final_score_{agent_id}"
                if metric_name not in episode.hist_data:
                    episode.hist_data[metric_name] = []
                episode.hist_data[metric_name].append(info["final_score"])
            if "action_reward_reduction" in info:
                metric_name = f"action_reward_reduction_{agent_id}"
                if metric_name not in episode.hist_data:
                    episode.hist_data[metric_name] = []
                episode.hist_data[metric_name].append(info["action_reward_reduction"])

                
    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger,
        result,
        **kwargs,
        ):

        main_rew = result[ENV_RUNNER_RESULTS]["hist_stats"]["policy_main_reward"] #.pop("policy_main_reward")

        # INFO: This only works with old rewards... 
        # For new rewards, one needs to figure out a better way to do this.
        # Unfortunately, the results lists will be of unequal length for the
        # different policies, as they are not participating in the same number
        # of games. One would need to figure out a way to reconstruct who played
        # against whom.

        winner_ids = result[ENV_RUNNER_RESULTS]["hist_stats"]["winner_ids"]
        n_main_policy_win = 0
        for winners in winner_ids:
            if self.main_policy_id in winners:
                n_main_policy_win += 1

        win_rate = n_main_policy_win / len(main_rew)

        if "win_rate" not in result[ENV_RUNNER_RESULTS]["hist_stats"]:
            result[ENV_RUNNER_RESULTS]["hist_stats"]["win_rate"] = []
        result[ENV_RUNNER_RESULTS]["hist_stats"]["win_rate"].append(win_rate)

        print(f"Iter={algorithm.iteration} win-rate={win_rate:3f} -> ", end="")

        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > self.win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played modules
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # agent_id = [0|1] -> policy depends on episode ID
                # This way, we make sure that both policies sometimes play
                # (start player) and sometimes agent1 (player to move 2nd).
                if agent_id == 0:
                    return "main"

                else:
                    return "main_v{}".format(
                        np.random.choice(list(range(1, self.current_opponent + 1)))
                    )

            main_policy = algorithm.get_policy("main")
            new_policy = algorithm.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(main_policy),
                policy_mapping_fn=policy_mapping_fn,
                config=main_policy.config,
                observation_space=main_policy.observation_space,
                action_space=main_policy.action_space,
            )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            main_state = main_policy.get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            algorithm.env_runner_group.sync_weights()

        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 3
        self.n_main_policy_win = 0


