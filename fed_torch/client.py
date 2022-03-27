from typing import Optional, Tuple
from collections import OrderedDict

import logging

import torch
import torch.nn as nn

from torch.optim import SGD
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
                a DataLoader object to get the train datasets
                (which must stay at client level)
            test_dataloader:
                a DataLoader object to get the train datasets
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
        parameters = weights_to_parameters(self.get_model_weights())
        return ParametersRes(parameters=parameters)

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

    def _fit_helper(self, epochs: int = 300, learning_rate: float = 0.001) -> int:
        """
        Helper method to train the PyTorch model at client level

        Args:
            epochs:
                number of epochs to iterate over
            learning_rate:
                learning rate of the Stochastic Gradient Descend
                optimizer

        Returns:
            the number of samples used for training
        """
        # loss criterion
        criterion = nn.CrossEntropyLoss()
        # stochastic gradient descend with fixed momentum
        # and given learning rate
        optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        self.model.to(self.device)
        self.model.train()
        for epoch in range(epochs):
            num_examples: int = 0
            epoch_loss: float = 0.0
            for data in self.train_dataloader:
                # extract data
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                # increase the number of examples
                # (useful for the FitRes returned
                # by the fit method from flower's
                # API))
                num_examples += len(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()
                # perform a propagation pass
                # in the neural network model
                labels_hat = self.model(inputs)
                # compute the loss
                loss = criterion(labels_hat, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            logger.info(f"Epoch: {epoch}/{epochs}, Loss: {epoch_loss}")
        # retrieve total number of training samples
        return num_examples

    def fit(self, ins: FitIns) -> FitRes:
        """
        Overrides the base class abc method
        Args:
            ins: a fit instance from flower
                library
                NOTICE: a flower FitIns has
                a `config` dictionary with
                the useful parameters for the
                fit process

        Returns:
            an instance of type FitRes from
            flower library
        """
        logger.info(f"[Client {self.id}] training...")
        # parsing fit configurations
        configs = ins.config
        epochs = int(configs["epochs"])
        learning_rate = float(configs["learning_rate"])

        # set model parameters
        weights = parameters_to_weights(ins.parameters)
        self.set_model_weights(weights)

        # fit the model
        num_examples = self._fit_helper(epochs, learning_rate)

        # get post training parameters
        post_training_weights = self.get_model_weights()
        post_training_params = weights_to_parameters(post_training_weights)

        # return the fit result (FitRes object)
        return FitRes(
            parameters=post_training_params, num_examples=num_examples  # noqa
        )

    def _eval_helper(self) -> Tuple[float, float, int]:
        """
        Helper for eval to obtain the final loss and the
        final accuracy. Adapted mainly from examples
        available in the official torch documentation

        Returns:
            a tuple containing final loss and accuracy
            expressed as floats and the number of test
            samples
        """
        # loss criterion
        criterion = nn.CrossEntropyLoss()
        loss: float = 0.0
        correct: int = 0
        test_size: int = 0
        num_examples: int = 0
        with torch.no_grad():
            for data in self.test_dataloader:
                # extract data
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                # update number of test examples
                # (needed by the EvaluateRes returned
                # by the evaluate method from flower's
                # API)
                num_examples += len(inputs)
                # perform a step in the net
                labels_hat = self.model(inputs)
                # compute the loss
                loss += criterion(labels_hat, labels).item()
                _, predictions = torch.max(labels_hat.data, 1)
                # update the number of predictions
                test_size += labels.size(0)
                # update the number of correctly predicted labels
                correct += (predictions == labels).sum().item()

        return loss, correct / test_size, num_examples

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the provided weights using the locally held datasets.
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
        loss, accuracy, num_examples = self._eval_helper()

        logger.info(f"[Client {self.id}] Loss: {loss} Accuracy: {accuracy}")
        return EvaluateRes(
            loss=loss,
            num_examples=num_examples,
            metrics={"metrics": accuracy},  # must be a dictionary
        )
