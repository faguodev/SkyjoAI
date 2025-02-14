import functools
import json
import logging
import os

import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from callback_functions import (RewardDecay_Callback,
                                SkyjoLogging_and_SelfPlayCallbacks)
from custom_models.action_mask_model import TorchActionMaskModel
from custom_models.fixed_policies import (PreProgrammedPolicyOneHot,
                                          PreProgrammedPolicySimple)
from environment.skyjo_env import env as skyjo_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

skyjo_config = {
    "num_players": 3,
    "reward_config": {
        "score_penalty": 2.0, # Seems useless
        "reward_refunded": 10,
        "final_reward": 100,
        "score_per_unknown": 5.0,
        "action_reward_decay": 1.0,
        "old_reward": False,
        "curiosity_reward": 4.0,
    },
    "observe_other_player_indirect": True,
    "render_mode": "human",
    "observation_mode": "onehot",
}

model_config = {
    "custom_model": TorchActionMaskModel,
    'vf_share_layers': True,
    # Add the following keys:
    "fcnet_hiddens": [2048, 2048, 1024, 512],
    "fcnet_activation": "relu",
}

def env_creator(config):
    return PettingZooEnv(skyjo_env(**config))

register_env("skyjo", env_creator)

test_env = env_creator(skyjo_config)
obs_space = test_env.observation_space
act_space = test_env.action_space

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if 0 == agent_id:
        return "main"
    elif 1 == agent_id:
        return "policy_1"
    else:
        return "policy_2"

config = (
    PPOConfig()
    .training(model=model_config, )
    .environment("skyjo", env_config=skyjo_config)
    .framework('torch')
    .callbacks(functools.partial(
            SkyjoLogging_and_SelfPlayCallbacks,
            win_rate_threshold=0.8,
            action_reward_reduction=0.2,
        )
    )
    #.callbacks(RewardDecayCallback)
    .env_runners(num_env_runners=5)
    .rollouts(num_rollout_workers=5, num_envs_per_worker=1)
    .resources(num_gpus=1)
    .multi_agent(
        policies={
            "main": (None, obs_space[0], act_space[0], {"entropy_coeff":0.03}),
            "policy_1": (PreProgrammedPolicyOneHot, obs_space[1], act_space[1], {"entropy_coeff":0.03}),
            "policy_2": (PreProgrammedPolicyOneHot, obs_space[2], act_space[2], {"entropy_coeff":0.03}),
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

config = "_others_direct"
model_save_dir = "trained_models/v6_trained_models_new_rewards" + config
os.makedirs(model_save_dir, exist_ok=True)
max_steps = 1e10
max_iters = 100000

#algo.restore("v0_pre_trained_model_others_direct_old_rewards/checkpoint_800")

#region Training

if not os.path.exists("logs/logs6"):
    os.mkdir("logs/logs6")

for iters in range(max_iters):
    result = algo.train()

    # Can be adjusted as needed
    if iters % 1 == 0:
        with open(f"logs/logs6/result_iteration_{iters}.json", "w") as f:
            json.dump(result, f, indent=4, default=convert_to_serializable)

    if iters % 10 == 0:
        checkpoint_dir = model_save_dir + f"/checkpoint_{iters}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        algo.save(checkpoint_dir)
    if result["timesteps_total"] >= max_steps:
        print(f"training done, because max_steps {max_steps} {result['timesteps_total']} reached")
        break
else:
    print(f"training done, because max_iters {max_iters} reached")

final_dir = model_save_dir + f"/final"
os.makedirs(final_dir, exist_ok=True)
algo.save(final_dir)

#endregion