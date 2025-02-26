import functools
import json
import logging
import os
import glob
from typing import Optional

import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.policy import Policy

from callback_functions import (RewardDecay_Callback, SkyjoLogging_and_SelfPlayCallbacks)
from custom_models.action_mask_model import TorchActionMaskModel
from custom_models.fixed_policies import (RandomPolicy, PreProgrammedPolicyEfficientOneHot, PreProgrammedPolicyOneHot, PreProgrammedPolicySimple)
from environment.skyjo_env import env as skyjo_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#You can either load a policy to continue training it or train a completely new model!

# Load configuration
#----------------------------Load model config from grid_search------------------------------
model_path = "obs_simple_indirect_True_vf_True_cr_5_ar_1_decay_1_ent_0.01_nn_[2048, 2048, 1024, 512]"
safe_path = model_path
config_path = f"logs/grid_search/{model_path}/experiment_config.json"

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
#---------------------------------------------------------------------------------------------------------


#----------------------------Load model config from previous self_play------------------------------
#model_path = "obs_eff_one_hot_indirect_True_vf_True_cr_5_ar_1_decay_0.996_ent_0.03_nn_[64, 64]"
#safe_path = "obs_eff_one_hot_indirect_True_vf_True_cr_5_ar_1_decay_0.996_ent_0.03_nn_[64, 64]"

    #--------------if file "experiment_config" is available use this------------------
#config_path = f"logs/self_play/{model_path}/experiment_config.json"

#with open(config_path, "r") as f:
#    config = json.load(f)

# Extract configuration parameters
#observation_mode = config["observation_mode"]
#observe_other_player_indirect = config["observe_other_player_indirect"]
#vf_share_layers = config["vf_share_layers"]
#curiosity_reward = config["curiosity_reward"]
#action_reward_reduction = config["action_reward_reduction"]
#action_reward_decay = config["action_reward_decay"]
#entropy_coeff = config["entropy_coeff"]
#neural_network_size = config["neural_network_size"]
    #-----------------------------------------------------------------------------------

    #--------------if no file "experiment_config" is available adapt the following manually------------------
#----------------Config for training completely new model (or if no "experiment_config" file is available)-----------------
#safe_path = "obs_eff_one_hot_indirect_True_vf_True_cr_0_ar_1_decay_0.95_ent_0.03_nn_[64]"

#observation_mode = "efficient_one_hot"
#observe_other_player_indirect = True
#vf_share_layers = True
#curiosity_reward = 0 
#action_reward_reduction = 1
#action_reward_decay = 0.95
#entropy_coeff = 0.03
#neural_network_size = [64] 
#---------------------------------------------------------------------------------------------------------


def get_latest_policy_path(base_path: str) -> Optional[str]:
    """Find the latest policy path either in final/ or checkpoint_x/ directories."""
    # Check final directory first
    final_policy_path = os.path.join(base_path, "final", "policies", "main")
    if os.path.exists(final_policy_path):
        return final_policy_path
    
    # Use os.listdir instead of glob
    try:
        all_dirs = os.listdir(base_path)
        checkpoint_dirs = [d for d in all_dirs if d.startswith("checkpoint_")]
        
        if not checkpoint_dirs:
            print(f"No checkpoints found in {base_path}")
            return None
        
        # Get the checkpoint with highest number
        latest_checkpoint = max(
            checkpoint_dirs,
            key=lambda x: int(x.split("checkpoint_")[-1])
        )
        
        latest_policy_path = os.path.join(base_path, latest_checkpoint, "policies", "main")
        if not os.path.exists(latest_policy_path):
            print(f"No policy found at {latest_policy_path}")
            return None
            
        print(f"Found latest policy at: {latest_policy_path}")
        return latest_policy_path
        
    except Exception as e:
        print(f"Error accessing directory {base_path}: {e}")
        return None
    
# Environment configuration
skyjo_config = {
    "num_players": 2,
    "reward_config": {
        "reward_refunded": 10,
        "final_reward": 100,
        "score_per_unknown": 5.0,
        "action_reward_reduction": 1.0,
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

# Create directories for logging and checkpoints
os.makedirs(f"logs/self_play/{safe_path}", exist_ok=True)
os.makedirs(f"trained_models/self_play/{safe_path}", exist_ok=True)


#---------------------------EITHER automatically choose model to load from previous training - NOT needed if new model is trained------------------------------
# Load the latest trained policy

    #----------------------from grid_search------------------
latest_policy_path = get_latest_policy_path(
    f"trained_models/grid_search/{model_path}"
)
    #------------------------------------------------------


    #----------------------from self_play------------------
#latest_policy_path = get_latest_policy_path(
#    f"trained_models/self_play/{model_path}"
#)
    #------------------------------------------------------


if not latest_policy_path:
    raise ValueError("No trained policy found!")
#---------------------------------------------------------------------------------------------------------


#------------------------OR manually chosen checkpoint to load model from - NOT needed if new model is trained----------------------
  #checkpoint to load from
#checkpoint = "checkpoint_5000"
#latest_policy_path = "trained_models/self_play/" + model_path + f"/{checkpoint}" + "/policies/main"
#---------------------------------------------------------------------------------------------------------

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
        )
    )
    .env_runners(num_env_runners=5)
    .rollouts(num_rollout_workers=5, num_envs_per_worker=1)
    .resources(num_gpus=1)
    .multi_agent(
        policies={
            "main": (None, obs_space[0], act_space[0], {"entropy_coeff": entropy_coeff}),
            "policy_1": (RandomPolicy, obs_space[1], act_space[1], {"entropy_coeff": entropy_coeff}),  #choose policy to train against: eg RandomPolicy, or a smart Pre-Programmed policy (depending on the observation-scheme:
                                                                                                                #PreProgrammedPolicyEfficientOneHot, PreProgrammedPolicyOneHot, PreProgrammedPolicySimple)
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


#----------------------load pre-trained model - NOT needed if new model is trained------------------------
restored_policy_weights = Policy.from_checkpoint(latest_policy_path).get_weights()
# Restore the latest policy
print(restored_policy_weights)
algo.set_weights({"main": restored_policy_weights})
#---------------------------------------------------------------------------------------------------------


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

# Training loop
max_iters = 100000   #adapt training iterations here
max_steps = 1e10     #adapt number of total environment steps sampled here

#safe configuration
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

with open(f"logs/self_play/{safe_path}/experiment_config.json", "w") as f:
    json.dump(config_params, f, indent=4, default=convert_to_serializable)
    print("safed config")

#TRAINING
for iters in range(max_iters):
    result = algo.train()
    
    # Log results every iteration
    if iters % 1 == 0:
        log_path = f"logs/self_play/{safe_path}/result_iteration_{iters}.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=4, default=convert_to_serializable)
    
    # Save checkpoint every 250 iterations
    if iters % 250 == 0:
        checkpoint_dir = f"trained_models/self_play/{safe_path}/checkpoint_{iters}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        algo.save(checkpoint_dir)
    
    if result["timesteps_total"] >= max_steps:
        print(f"Training done, because max_steps {max_steps} {result['timesteps_total']} reached")
        break
else:
    print(f"Training done, because max_iters {max_iters} reached")

# Save final model
final_dir = f"trained_models/self_play/{safe_path}/final"
os.makedirs(final_dir, exist_ok=True)
algo.save(final_dir)