from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np

from .utils import check_dataset_name, create_local_dataset
from ..local_dataset import LocalDataset


class NumberClientsError(Exception):
    """ error presents in the DistributeDataset class"""

    def __init__(self, n_clients):
        self.clients = n_clients
        self.message = f"Number of clients must be at least 1, got {n_clients}"
        super().__init__(self.message)


class DivergenceError(Exception):
    """ error presents in the DistributeDataset class"""

    def __init__(self, divergence):
        self.divergence = divergence
        self.message = f"divergence must be between 0 and 1, got {divergence}"
        super().__init__(self.message)


class DistributeDataset(ABC):

    def __init__(
            self,
            dataset: str,
            download_path: str,
            n_clients: int,
            download=True,

    ):
        """
        Args:
            dataset:
                String, containing the datasets ['CIFAR10', 'CIFAR100', 'MNIST']
            download_path:
                String, path for the datasets
            download:
                Boolean, to download the datasets or not
        """

        # if you're on Windows and you have problems downloading
        # the dataset, uncomment the following two lines
        # import ssl
        # ssl._create_default_https_context = ssl._create_unverified_context

        dataset = dataset.upper()

        self.dataset = dataset  # name of the dataset
        self.divergence = None  # divergence for distribute the dataset
        self.client_test = None  # list of test client datasets
        self.client_train = None  # list of train client datasets
        if n_clients < 0:
            raise NumberClientsError(n_clients)

        self.n_clients = n_clients

        download_dataset, local_dataset = check_dataset_name(dataset)
        self.local_dataset = local_dataset  # torch.dataset for the dataset name

        self.train = download_dataset(root=download_path, train=True, download=download)
        self.test = download_dataset(root=download_path, train=False, download=download)

        self.size_class_train = len(self.train) // len(self.train.classes)
        self.size_class_test = len(self.test) // len(self.test.classes)

    def get_train_test(self) -> Tuple[LocalDataset, LocalDataset]:
        """
        It returns the entire training and test dataset
        Returns:
            the train and test torch dataset
        """

        train = self.local_dataset(
            self.train.data,
            self.train.targets
        )

        test = self.local_dataset(
            self.test.data,
            self.test.targets
        )
        return train, test

    def divide_dataset(self,
                       divergence: float,
                       ) -> Tuple[List[LocalDataset], List[LocalDataset]]:
        """
        divide the datasets according to the divergence.
        - 0 means to have (mostly) a uniform distribution of the classes among clients
        - 1 means to have (mostly) a single class present in each class
        mostly because the length of the dataset is not always divisible by the n_clients.

        Args:
            divergence:
                Float, how spread are the classes [0,1]
        Returns:
            the list of train and test local dataset for each client

        """
        if not 0 <= divergence <= 1:
            raise DivergenceError(divergence)

        self.divergence = divergence

        # split the dataset according to the divergence
        class_clients = self.get_index(divergence)

        # creates the local datasets
        client_train = create_local_dataset(
            class_clients[0],
            self.train.data, np.array(self.train.targets),
            self.local_dataset
        )

        client_test = create_local_dataset(
            class_clients[1],
            self.test.data, np.array(self.test.targets),
            self.local_dataset
        )
        self.client_train = client_train
        self.client_test = client_test

        return client_train, client_test

    def get_index(self,
                  divergence: float = None
                  ):
        """
        Calculates the indexes for each client. The indexes are then used to sample
        from the entire dataset.
        Args:
            divergence:
                the "beta" parameter of the Beta distributions
        Returns:
            class_client_mat:
                shape=(self.train.classes, self.n_client, class_elements_per_client)
        """
        # get the sorted index of the classes
        # [0,0,..0, 1,1,...,1, 2,2...9]
        sort_classes_train = np.argsort(self.train.targets)
        sort_classes_test = np.argsort(self.test.targets)

        class_client_train = self.strategy(
            self.size_class_train // self.n_clients,
            sort_classes_train,
            len(self.train.classes),
            self.size_class_train,
            self.n_clients,
            divergence,
            # TODO check: not all class in training
        )
        class_client_test = self.strategy(
            self.size_class_test // self.n_clients,
            sort_classes_test,
            len(self.test.classes),
            self.size_class_test,
            self.n_clients,
            divergence,
            )

        return class_client_train, class_client_test

    @abstractmethod
    def strategy(self, *args):
        pass
