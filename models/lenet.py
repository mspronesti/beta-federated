import torch
import torch.nn as nn
from flwr.common import Weights
from collections import OrderedDict


class LeNet(nn.Module):
    def __init__(self, in_features=3, num_classes=10):
        """
        Constructs a LeNet Convolutional Neural Network Model
        using pytorch

        Args:
            in_features:
                number of input features of the data
            num_classes:
                number of classes of the data
        """
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_features, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.f1 = nn.Linear(16 * 5 * 5, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Performs a forward step in the neural network"""
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = self.act(self.f1(x))
        x = self.act(self.f2(x))
        return self.act(self.f3(x))

    def get_weights(self) -> Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
