import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics import (ENV_RUNNER_RESULTS)

class RewardDecay_Callback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Decay the reward scaling factor over training iterations
        action_reward_decay = max(0.05, 1.0 - result["training_iteration"] * 0.005)
        # env = algorithm.workers.local_worker().env
        # env = algorithm.workers.local_env_runner.env
        algorithm.config.env_config["action_reward_decay"] = action_reward_decay
        #logger.info(action_reward_decay)

class SkyjoLogging_and_SelfPlayCallbacks(DefaultCallbacks):

    def __init__(self, win_rate_threshold, action_reward_reduction):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
        self.winning_policy = []
        self.win_rate_threshold = win_rate_threshold
        self.action_reward_reduction = action_reward_reduction

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
    
            if info is not None and "final_card_sum" in info:
                metric_name = f"final_card_sum_{agent_id}"
                episode.custom_metrics[metric_name] = info["final_card_sum"]
            if info is not None and "n_hidden_cards" in info:
                metric_name = f"n_hidden_cards_{agent_id}"
                episode.custom_metrics[metric_name] = info["n_hidden_cards"]

                #episode.custom_metrics["winning_policy"].append([episode.policy_for(id) for id in info["winner_ids"]])
                
    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger,
        result,
        **kwargs,
        ):

        #for p_id in self.playing_polices:
        #    print("Policies playing: ", p_id)

        
        #opponent_rew_2 = result[ENV_RUNNER_RESULTS]["hist_stats"].pop(f"policy_{self.playing_polices[2]}_reward")
        main_rew = result[ENV_RUNNER_RESULTS]["hist_stats"]["policy_main_reward"] #.pop("policy_main_reward")
        won = 0
        #n_games = len(main_rew)
        #for rew in main_rew:
        #    if rew%1 != 0:
        #        won += 1

        #win_rate = won/ len(main_rew)

        # INFO: This only works with old rewards... 
        # For new rewards, one needs to figure out a better way to do this.
        # Unfortunately, the results lists will be of unequal length for the
        # different policies, as they are not participating in the same number
        # of games. One would need to figure out a way to reconstruct who played
        # against whom.

        for r_main in main_rew:
            if r_main == 100.0:
                won += 1

        win_rate = won / len(main_rew)

        
        print(f"Iter={algorithm.iteration} win-rate={win_rate:3f}, reward_decay={algorithm.config.env_config['reward_config']['action_reward_decay']:3f} -> ", end="")

        action_reward_reduction = max(0.01, algorithm.config.env_config['reward_config']["action_reward_decay"] * 0.98)
        algorithm.config.env_config['reward_config']["action_reward_decay"] = action_reward_reduction

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

            algorithm.config.env_config['reward_config']['action_reward_decay'] = self.action_reward_reduction
        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 3

        #print(f"Matchups:\n{self._matching_stats}")

