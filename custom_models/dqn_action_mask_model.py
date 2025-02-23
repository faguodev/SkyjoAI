import time

from gymnasium.spaces import Dict
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
import numpy as np

torch, nn = try_import_torch()


class DQNActionMaskModel(TorchModelV2, nn.Module):
    """
    DQN model with action masking support.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observations" in orig_space.spaces
        )

        # with open(f"/home/henry/Documents/SharedDocuments/Uni/TU/3.Semester/AdvRL/SkyjoAI/file{time.time()}.txt", "w") as f:
            # f.write("Init")

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        self.num_actions = action_space.n
        assert num_outputs == self.num_actions, f"Expected num_outputs={self.num_actions}, got {num_outputs}"

        self.q_network = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_q_network",
        )

        self.no_masking = model_config.get("custom_model_config", {}).get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        q_values, _ = self.q_network({"obs": input_dict["obs"]["observations"]})

        if self.no_masking:
            return q_values, state

        # Ensure action_mask and q_values have the same shape
        assert action_mask.shape[1] == q_values.shape[1], f"Shape mismatch: {action_mask.shape} vs {q_values.shape}"

        # Apply action mask safely
        # inf_mask = torch.where(action_mask.bool(), torch.zeros_like(q_values), torch.full_like(q_values, FLOAT_MIN))
        # masked_q_values = q_values + inf_mask


        # with open(f"/home/henry/Documents/SharedDocuments/Uni/TU/3.Semester/AdvRL/SkyjoAI/file{time.time()}.txt", "w") as f:
        #     f.write(q_values)
        #     f.write(self.internal_model({'obs': input_dict['obs']['observations']}))
        print(f"{q_values=}")
        # print(self.internal_model({'obs': input_dict['obs']['observations']}))
        # print(f"{action_mask=}")

        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN*0.1)
        masked_q_values = q_values + inf_mask
        # print(f"{inf_mask=}")
        # print(f"{masked_q_values=}")
        print(f"Probabilities: {torch.softmax(masked_q_values, dim=-1)}")

        return masked_q_values, state

    def get_q_values(self, input_dict):
        q_values, _ = self.q_network({"obs": input_dict["obs"]["observations"]})
        return q_values