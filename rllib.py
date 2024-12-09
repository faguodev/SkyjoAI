from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from environment.skyjo_env import env as skyjo_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import logging
import json
import numpy as np

from models.action_mask_model import TorchActionMaskModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RewardDecayCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Decay the reward scaling factor over training iterations
        action_reward_decay = max(0.05, 1.0 - result["training_iteration"] * 0.01)
        env = algorithm.workers.local_worker().env
        env.update_action_reward_decay(action_reward_decay)
        logger.info(action_reward_decay)


skyjo_config = {
    "num_players": 3,
    "score_penalty": 2.0,
    "observe_other_player_indirect": True,
    "mean_reward": 1.0,
    "reward_refunded": 10,
    "final_reward": 100,
    "score_per_unknown": 5.0,
    "render_mode": "human",
}

model_config = {
    'custom_model': TorchActionMaskModel 
}

def env_creator(config):
    return PettingZooEnv(skyjo_env(**config))

register_env("skyjo", env_creator)

test_env = env_creator(skyjo_config)
obs_space = test_env.observation_space
act_space = test_env.action_space

def policy_mapping_fn(agent_id, _):
    return agent_id

config = (
    PPOConfig()
    .training(model=model_config)
    .environment("skyjo", env_config=skyjo_config)
    .framework('torch')
    .training(model=model_config)
    #.callbacks(RewardDecayCallback)
    .env_runners(
        num_env_runners=1,
    )
    # .multi_agent(
    #     policies={
    #         "player_0": (None, obs_space, act_space, {}),
    #         "player_1": (None, obs_space, act_space, {}),
    #         "player_2": (None, obs_space, act_space, {})
    #     },
    #     policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
    # )
    .evaluation(evaluation_num_env_runners=0)
    .debugging(log_level="INFO")
)

algo = config.build()


def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable types."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

for i in range(1000):
    result = algo.train()
    result.pop("config")
    print(result)

    if i % 5 == 0:
        # Save result to JSON file
        with open(f"result_iteration_{i}.json", "w") as f:
            json.dump(result, f, indent=4, default=convert_to_serializable)
