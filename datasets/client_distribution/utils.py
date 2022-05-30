from enum import Enum

from torchvision import datasets
from ..cifar_local_dataset import CifarLocalDataset
from ..mnist_local_dataset import MnistLocalDataset
from ..local_dataset import LocalDataset

import numpy as np


class DatasetNameError(Exception):
    def __init__(self, dataset):
        self.name = dataset
        self.message = f"Unknown dataset {dataset}. " \
                       f"Valid values are CIFAR10, CIFAR100 and MNIST",

        super().__init__(self.message)


class BuildDataset(Enum):
    CIFAR10 = (datasets.CIFAR10, CifarLocalDataset)
    CIFAR100 = (datasets.CIFAR100, CifarLocalDataset)
    MNIST = (datasets.MNIST, MnistLocalDataset)

    @classmethod
    def has_value(cls, value):
        if value not in [item[0] for item in cls.__members__.items()]:
            raise DatasetNameError(value)


def check_dataset_name(dataset):
    BuildDataset.has_value(dataset.upper())
    return BuildDataset[dataset.upper()].value


def create_local_dataset(
        class_clients_mat: np.array,
        data: np.array,
        labels: np.array,
        local_dataset
):
    """
    Args:
        class_clients_mat:
            matrix shape(n_classes, n_clients, num_sample)
        data:
            it is the entire dataset (len_dataset, h, w, channel)
        labels:
            numpy array (num_labels, 1)
        local_dataset (Any):
            the dataset class used in the training
    """
    # row clients, column labels
    class_clients_mat = class_clients_mat.transpose()
    client_datasets = []
    for j, client in enumerate(class_clients_mat):
        client_indexes = np.hstack(client).astype(int)

        unique_labels_count = np.unique(labels[client_indexes])

        dataset = local_dataset(
            data[client_indexes],
            labels[client_indexes],
            unique_labels_count,
            client_id=j
        )

        client_datasets.append(dataset)

    return client_datasets
