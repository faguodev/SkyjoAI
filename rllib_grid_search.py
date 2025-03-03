import functools
import itertools
import json
import logging
import os
import numpy as np
from collections import defaultdict
import re
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from callback_functions import SkyjoLogging_and_SelfPlayCallbacks
from custom_models.action_mask_model import TorchActionMaskModel
from custom_models.fixed_policies import PreProgrammedPolicyOneHot, PreProgrammedPolicySimple, SingleAgentPolicy, RandomPolicy, EfficientSingleAgentPolicy, PreProgrammedPolicyEfficientOneHot
from environment.skyjo_env import env as skyjo_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Change this for your own setup

defaults = {
    "observation_mode": "simple_port_to_other",
    "observe_other_player_indirect": False,
    "vf_share_layers": True,
    "curiosity_reward": 0,
    "action_reward_reduction": 1,
    "action_reward_decay": 0.98,
    "entropy_coeff": 0.01,
    "neural_network_size": [32],
    "max_iters": 1000,
}

# Search spaces
tuning_stages = [
    #{"observation_mode": ["simple_port_to_other", "efficient_one_hot_port_to_other"]},
    #{"curiosity_reward": [0.0, 5]},
    {"action_reward_decay": [0.98, 0.995, 0.95], "action_reward_reduction": [1, 5]},
    {
        "neural_network_size": [
            [16], 
            [32],
            [64],
        ], 
        "observation_mode": ["simple_port_to_other","efficient_one_hot_port_to_other"],
        "max_iters": [1000],   
    },
    {
        "neural_network_size": [
            [32, 32],
            [64, 64],
        ], 
        "observation_mode": ["simple_port_to_other","efficient_one_hot_port_to_other"],
        "max_iters": [5000],
    },
]

def train_model(
        observation_mode, 
        observe_other_player_indirect, 
        vf_share_layers, 
        curiosity_reward, 
        action_reward_reduction, 
        action_reward_decay, 
        entropy_coeff, 
        max_iters,
        #learning_rate, 
        neural_network_size):

    skyjo_config = {
        "num_players": 2,
        "reward_config": {
            "reward_refunded": 10,
            "final_reward": 100,
            "score_per_unknown": 5.0,
            "action_reward_reduction": action_reward_reduction, # Search
            "old_reward": False,
            "curiosity_reward": curiosity_reward, # Search
        },
        "observe_other_player_indirect": observe_other_player_indirect, # 1.Search
        "render_mode": "human",
        "observation_mode": observation_mode, # 1.Search
    }

    if observation_mode == "onehot":
        opponent_policy = PreProgrammedPolicyOneHot
    elif observation_mode == "simple":
        opponent_policy = PreProgrammedPolicySimple
    elif observation_mode == "simple_port_to_other":
        opponent_policy = SingleAgentPolicy
    elif observation_mode == "efficient_one_hot_port_to_other":
        opponent_policy = EfficientSingleAgentPolicy
    elif observation_mode == "efficient_one_hot":
        opponent_policy = PreProgrammedPolicyEfficientOneHot

    model_config = {
        "custom_model": TorchActionMaskModel,
        'vf_share_layers': vf_share_layers, # 2. Search
        # Add the following keys:
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

    config = (
        PPOConfig()
        .training(model=model_config)#, lr=learning_rate)
        .environment("skyjo", env_config=skyjo_config)
        .framework('torch')
        .callbacks(functools.partial(
                SkyjoLogging_and_SelfPlayCallbacks,
                main_policy_id=0,
                win_rate_threshold=0.8,
            )
        )
        #.callbacks(RewardDecayCallback)
        .env_runners(num_env_runners=4)
        .rollouts(num_rollout_workers=4, num_envs_per_worker=1)
        .resources(num_gpus=1)
        .multi_agent(
            policies={
                "main": (None, obs_space[0], act_space[0], {"entropy_coeff": entropy_coeff}),
                "policy_1": (opponent_policy, obs_space[1], act_space[1], {}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
        )
        .evaluation(evaluation_num_env_runners=0)
        .debugging(log_level="INFO")
        .api_stack(
            enable_rl_module_and_learner=False,
        )
        .learners(num_gpus_per_learner=1)
    )

    algo = config.build()

    #region Logging
    
    # Automatically generate a unique directory name
    param_string = f"obs_{observation_mode}_indirect_{observe_other_player_indirect}_vf_{vf_share_layers}_cr_{curiosity_reward}_ar_{action_reward_reduction}_fixed_decay_{action_reward_decay}_ent_{entropy_coeff}_nn_{neural_network_size}_against_other"#_lr_{learning_rate}

    def convert_to_serializable(obj):
        """Convert non-serializable objects to serializable types."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return "error"

    #endregion

    model_save_dir = f"trained_models/grid_search/{param_string}"
    os.makedirs(model_save_dir, exist_ok=True)

    #region Training
    
    logs_save_dir = f"logs/grid_search/{param_string}"
    os.makedirs(logs_save_dir, exist_ok=True)

    # Save configuration parameters
    config_params = {
        "observation_mode": observation_mode,
        "observe_other_player_indirect": observe_other_player_indirect,
        "vf_share_layers": vf_share_layers,
        "curiosity_reward": curiosity_reward,
        "action_reward_reduction": action_reward_reduction,
        "action_reward_decay": action_reward_decay,
        "entropy_coeff": entropy_coeff,
        #"learning_rate": learning_rate,
        "neural_network_size": neural_network_size,
    }
    
    with open(f"{logs_save_dir}/experiment_config.json", "w") as f:
        json.dump(config_params, f, indent=4, default=convert_to_serializable)

    tmp_action_reward_reduction = action_reward_reduction

    for iters in range(max_iters):
        print(f"Starting iteration {iters}/{max_iters}")
        result = algo.train()
        tmp_action_reward_reduction *= action_reward_decay
        if tmp_action_reward_reduction < 0.05:
            tmp_action_reward_reduction = 0
        algo.env_runner_group.foreach_env(lambda env: env.env.update_action_reward_reduction(tmp_action_reward_reduction))

        # Can be adjusted as needed
        if iters % 1 == 0:
            with open(f"{logs_save_dir}/result_iteration_{iters}.json", "w") as f:
                json.dump(result, f, indent=4, default=convert_to_serializable)

        if iters % 250 == 0:
            checkpoint_dir = f"{model_save_dir}/checkpoint_{iters}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            algo.save(checkpoint_dir)
    
    final_dir = f"{model_save_dir}/final"
    os.makedirs(final_dir, exist_ok=True)
    algo.save(final_dir)

    return param_string

def evaluate(logs_path):
    files = sorted([os.path.join(logs_path, f) for f in os.listdir(logs_path) 
                   if os.path.isfile(os.path.join(logs_path, f)) and f.startswith('result_iteration_')],
                   key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    hist_stats = defaultdict(list)
    for file in files:
        with open(file) as f:
            d = json.load(f)
            for hist_stat in ["final_score_0", "win_rate"]:
                hist_stats[hist_stat].append(np.array(d["env_runners"]["hist_stats"][hist_stat]))
                hist_stats[hist_stat+"_mean"].append(np.mean(hist_stats[hist_stat][-1]))
    
    # Find the window with the highest win rate
    win_rate_means = hist_stats["win_rate_mean"]
    max_win_rate = 0
    max_win_rate_idx = 0
    
    for idx in range(len(win_rate_means) - 30):
        current_win_rate = np.mean(win_rate_means[idx:idx+30])
        if current_win_rate > max_win_rate:
            max_win_rate = current_win_rate
            max_win_rate_idx = idx
    
    mean_win_rate = np.mean(hist_stats["win_rate_mean"][max_win_rate_idx:max_win_rate_idx+30])
    print(f"Mean win rate: {mean_win_rate}")
    
    return mean_win_rate# mean_final_score, 


def evaluate_candidate(params):
    """
    Evaluate a candidate parameter set by training a model and evaluating its performance.
    Returns the mean score from the last 10 iterations.
    """
    param_string = train_model(**params)
    logs_path = f"logs/grid_search/{param_string}"
    return evaluate(logs_path)

def run_staged_grid_search():
    best_params = defaults.copy()

    print(f"Starting grid search with {len(tuning_stages)} stages")

    for i,stage in enumerate(tuning_stages):
        print(f"Running stage {i+1}/{len(tuning_stages)}")
        candidates = []
        for values in itertools.product(*stage.values()):
            new_params = best_params.copy()
            for key, value in zip(stage.keys(), values):
                new_params[key] = value
            candidates.append(new_params)
        
        # Train all candidates
        print(f"Training {len(candidates)} candidates for stage {i+1}")
        
        # Evaluate each candidate and find the best one
        best_score = -1
        best_candidate = None
        
        for candidate in candidates:
            print(f"Training candidate: {candidate}")
            score = evaluate_candidate(candidate)
            print(f"Candidate {candidate} achieved score: {score}")
            if score > best_score:
                best_score = score
                best_candidate = candidate
            print(f"Best score after stage {i+1}: {best_score}")
            print(f"Best candidate after stage {i+1}: {best_candidate}")

            
        if best_candidate is None:
            print(f"No valid candidates found in stage {i+1}")
            return best_params
    
            
        best_params.update(best_candidate)
        print(f"Best parameters after stage {i+1}: {best_params}")
    return best_params

run_staged_grid_search()
