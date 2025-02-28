# SkyjoAI

## Overview
Welcome to the SkyjoAI repository! This repository contains our implementation of AI agents trained to play the card game "Skyjo." You will find everything needed to play against trained agents, train your own agents, and evaluate different agents. The repository is structured into several folders, each serving a specific purpose.

## Repository Structure
### Folders
- **`custom_models/`**: Contains a custom action mask model for PPO training, and predefined policies.
- **`environment/`**: Contains the game implementation of "Skyjo" and the environment for training agents.
- **`logs/`**: Stores training log files when training with `rllib_self_play.py` or `rllib_grid_search.py`.
- **`trained_models/`**: Stores trained models. Some pre-trained models are included.

### Important Files
- **`callback_functions.py`**: Contains logging and training callback functions.
- **`eval.py`**, **`evaluation.ipynb`**, **`eval_compare_with_others.py`**: Used for evaluating trained models.
- **`rllib_self_play.py`**, **`rllib_grid_search.py`**: Used to train models.
- **`skyjo_gui.py`**, **`skyjo_gui_compare_with_others.py`**: Play a game of Skyjo against trained models.
- **`ARL_Skyjo.pdf`** contains the report that was written as part of our course

## Installation
To use this repository, first install all dependencies listed in `environment.yml`:
```bash
conda env create -f environment.yml
```

## How to Play
To play against a trained AI agent in a graphical interface, run:
```bash
python skyjo_gui.py
```
This will launch a game where you can play against one or multiple trained models. To select different models, modify these variables:
- `model_path`: Set to the name of the folder containing the model.
- `checkpoint`: Set to the checkpoint from which the model should be loaded.

If there is no `experiment_config.json` file for the model, manually configure the following parameters:
- `observation_mode`: Defines the observation space.
- `observe_other_player_indirect`: Boolean flag for observation mode.
- `vf_share_layers`: Boolean flag for shared network layers.
- `neural_network_size`: List of integers defining neural network layers, e.g., `[32, 32]`.

## How to Train
To train a new model, you can use:
- `rllib_grid_search.py`: Runs a grid search to find optimal training parameters. Run it with:
  ```bash
  python rllib_grid_search.py
  ```
- `rllib_self_play.py`: Trains a model with fixed parameters or continues training an existing model.

Training configuration is defined in:
- `skyjo_config`: Environment settings.
- `model_config`: Model-specific settings.
- `config`: Training algorithm settings.

## How to Evaluate
Three files provide evaluation capabilities:
- **`evaluation.ipynb`**: Compares and plots training logs.
- **`eval.py`**: Lets policies play against each other and tracks win rates.
- **`skyjo_gui.py`**: Enables qualitative evaluation through gameplay.

Additional comparisons can be made using:
- **`eval_compare_with_others.py`**: Evaluates policies against agents trained in a different single-agent environment.
- **`skyjo_gui_compare_with_others.py`**: Enables human players to compete against those agents.

## The Environment
The environment consists of two classes:
- **`SkyjoGame`** (in `skyjo_game.py`): Implements the game logic.
- **`SimpleSkyjoEnv`** (in `skyjo_env.py`): Creates a training environment for agents using the AEC API from PettingZoo.

### Configuration Parameters
- **`num_players`**: Number of players (2-8 recommended).
- **`observe_other_player_indirect`**: Boolean flag for observation mode.
- **`render_mode`**: "human" or None (for rendering, legacy setting).
- **`observation_mode`**: Defines observation scheme (`simple`, `onehot`, etc.).
- **`reward_config`**: Dictionary containing:
  - `reward_refunded`: Reward for discarding a complete column.
  - `final_reward`: Reward for winning.
  - `score_per_unknown`: Score assigned to unknown cards.
  - `action_reward_reduction`: Multiplier for each-turn rewards.
  - `curiosity_reward`: Reward for revealing a new card.
  - `old_reward`: True for end-of-game reward, False for each-turn reward.

### Example Configuration
```python
skyjo_config = {
    "num_players": 2,
    "reward_config": {
        "reward_refunded": 10,
        "final_reward": 100,
        "score_per_unknown": 5.0,
        "action_reward_reduction": 1,
        "old_reward": False,
        "curiosity_reward": 5,
    },
    "observe_other_player_indirect": True,
    "render_mode": "human",
    "observation_mode": "simple",
}
```
