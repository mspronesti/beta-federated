import torch
import torch.nn as nn
from flwr.common import Weights
from collections import OrderedDict
from abc import ABC, abstractmethod


class ModelBase(nn.Module, ABC):
    """Base class for Pytorch models"""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Must have implemented this.")

    def get_weights(self) -> Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        # it's important they are sorted!
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
