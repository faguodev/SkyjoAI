from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from environment.skyjo_env import env as skyjo_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import logging

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
    .evaluation(evaluation_num_env_runners=0)
    .debugging(log_level="INFO")
)

algo = config.build()

for i in range(10):
    result = algo.train()
    result.pop("config")
    print(result)

    if i % 5 == 0:
        #checkpoint_dir = algo.save_to_path()
        print(f"Checkpoint happened")