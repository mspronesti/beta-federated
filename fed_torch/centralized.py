import hydra
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets import DistributeUniform
from models import CNN
from models import LeNet

from logger import logger
from enum import Enum


class ModelError(Exception):
    """ error presents in the centralized_run"""
    def __init__(self, name):
        self.name = name
        self.message = f'Implemented models are Lenet fro Cifar and CNN for mnist,' \
                       f'got as dataset {name}'
        super().__init__(self.message)


class BuildModel(Enum):
    CIFAR10 = LeNet()
    CIFAR100 = LeNet()
    MNIST = CNN()

    @classmethod
    def has_value(cls, value):
        if value not in [item[0] for item in cls.__members__.items()]:
            raise ModelError(value)


class CentralizedNet:
    def __init__(self,
                 model_name,
                 lr,
                 epochs,
                 momentum,
                 batch_size,
                 device,
                 dataset,
                 download_path):

        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.dataset = dataset

        maker_dataset = DistributeUniform(dataset,
                                          download_path,
                                          n_clients=1)

        train, test = maker_dataset.get_train_test()

        self.train_loader = DataLoader(
            train, batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = DataLoader(
            test, batch_size * 2, num_workers=4, pin_memory=True
        )

        BuildModel.has_value(dataset.upper())
        self.model = BuildModel[dataset.upper()].value

    def fit(self):
        criterion = CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(),
                              self.lr,
                              self.momentum
                              )
        self.model.to(self.device)

        for epoch in range(self.epochs):
            # for each epoch
            epoch_loss: float = 0.0
            size: int = 0
            correct_predictions: int = 0

            for i, data in enumerate(self.train_loader, 0):
                # for each batch
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                _, predictions = torch.max(outputs.data, 1)
                correct_predictions += (predictions == labels).sum().item()
                size += labels.size(0)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(self.train_loader)
            train_accuracy = correct_predictions / size

            # Evaluate
            with torch.no_grad():
                size: int = 0
                correct_predictions: int = 0
                val_loss: float = 0.0
                for i, data in enumerate(self.test_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    correct_predictions += (predictions == labels).sum().item()
                    size += labels.size(0)
                    loss = criterion(outputs, labels)

                    val_loss += loss

                val_loss /= len(self.test_loader)
                val_accuracy = correct_predictions / size

            logger.info(f"Epoch: {epoch}/{self.epochs}, "
                        f"Training: loss {epoch_loss}, acc {train_accuracy} "
                        f"Validation: loss {val_loss} acc {val_accuracy}")


@hydra.main(config_path="../config/", config_name="config.yaml")
def main(cfg):

    centralized = CentralizedNet(
        'Lenet',
        cfg.fed_torch.lr,
        cfg.fed_torch.epochs,
        cfg.fed_torch.momentum,
        cfg.fed_torch.batch_size,
        cfg.device,
        cfg.dataset,
        cfg.dataset_download_path
    )

    centralized.fit()


if __name__ == "__main__":
    main()
