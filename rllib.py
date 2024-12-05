from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from environment.skyjo_env import env as skyjo_env

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

def env_creator(config):
    return PettingZooEnv(skyjo_env(**config))

register_env("skyjo", env_creator)


config = (
    PPOConfig()
    .environment("skyjo", env_config=skyjo_config)
    .env_runners(
        num_env_runners=2,
    )
    .evaluation(evaluation_num_env_runners=1)
)

algo = config.build()

for i in range(10):
    result = algo.train()
    result.pop("config")
    print(result)

    if i % 5 == 0:
        checkpoint_dir = algo.save_to_path()
        print(f"Checkpoint saved in directory {checkpoint_dir}")