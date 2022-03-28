import torch
import torch.nn as nn
from flwr.common import Weights
from collections import OrderedDict


class CNN(nn.Module):
    def __init__(self, in_features: int = 3, num_classes=10):
        """
        Constructs a Convolutional Neural Network Model using
        pytorch

        Args:
            in_features:
                number of input features of the data
            num_classes:
                number of classes of the data
        """
        super(CNN, self).__init__()
        # first convolutional layer, with
        # - in_features input channels
        # - 32 output channels
        # - 5-dimensional convolutional kernel
        self.conv1 = nn.Conv2d(in_features, 32, 5)
        # second convolutional layer, with
        # - 32 input channel
        # - 64 output channel
        # - 5-dimensional convolutional kernel
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # activation function
        self.act = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        """Performs a forward step in the neural network"""
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        return self.fc2(x)

    def get_weights(self) -> Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
