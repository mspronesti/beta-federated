from typing import Optional
from collections import OrderedDict

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    weights_to_parameters,
    parameters_to_weights,
)

logger = logging.getLogger(__name__)


class TorchClient(fl.client.Client):
    """Torch Client class inheriting from Flower's
    abstract base class"""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: Optional[str] = "cpu",
    ):
        """
        Args:
            client_id:
                unique integer identifier of this client
            model:
                a torch-compatible model to be run from
                this client
            train_dataloader:
                a DataLoader object to get the train dataset
                (which must stay at client level)
            test_dataloader:
                a DataLoader object to get the train dataset
                (which must stay at client level)
            device:
                a string identifying a valid torch device.
                Admitted values: cpu, cuda,
        """
        self.id = client_id
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def get_parameters(self) -> ParametersRes:
        """
        Retrieves the current model parameters
        Overrides the base class abc method

        Returns:
            a ParameterRes object
            containing the model's weights
            parametrization
        """
        return ParametersRes(parameters=weights_to_parameters(self.get_model_weights()))

    def get_model_weights(self) -> fl.common.Weights:
        return [values.cpu().numpy() for _, values in self.model.state_dict().items()]

    def set_model_weights(self, weights: fl.common.Weights):
        """Set model weights from a flwr.common.Weights object"""
        # it's important they are sorted!
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict)

    def fit(self, ins: FitIns) -> FitRes:
        """
        Overrides the base class abc method
        Args:
            ins: a fit instance from flower
                library

        Returns:
            an instance of type FitRes from
            flower library
        """
        logger.info(f"[Client {self.id}] fitting...")
        # use train dataset here ...
        # TODO: to be implemented
        pass

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the provided weights using the locally held dataset.
        Overrides the base class abc method

        Args:
            ins: EvaluateIns instance from server.
                See flower Client abstract base class for more

        Returns: EvaluateRes instance.

        """
        logger.info(f"[Client {self.id}] evaluating...")

        # update model's weights
        weights = parameters_to_weights(ins.parameters)
        self.set_model_weights(weights)

        # TODO: to be implemented
        # use test dataset here ...
        pass
