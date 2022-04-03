from typing import Tuple, List

import numpy as np
import torchvision
from torch.utils.data import Dataset

from .cifar_local_dataset import CifarLocalDataset


class DistributeDataset:
    """
    Distributes the dataset among the clients according to the parameter
    'divergence'. Divergence = 0, it splits uniformly the datasets among clients.

    Attributes:
        train:
            the training datasets
        test:
            test datasets
        n_clients:
            number of clients
        divergence:
            how spread are the classes [0,1]
        client_range:
            array containing the x-range on beta distribution.
    """

    def __init__(
        self,
        dataset: str,
        download_path: str,
        download=True,
        n_clients=1,
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

        self.dataset = dataset
        self.divergence = None
        self.client_datasets_test = None
        self.client_datasets_train = None

        if n_clients < 0:
            raise RuntimeError("Number of clients must be at least 1")

        self.n_clients = n_clients
        self.client_range = np.linspace(0, 1, n_clients + 1)

        if dataset == "CIFAR10":
            download_dataset = torchvision.datasets.CIFAR10

        elif dataset == "CIFAR100":
            download_dataset = torchvision.datasets.CIFAR100

        elif dataset == "MNIST":
            download_dataset = torchvision.datasets.MNIST

        else:
            raise ValueError(
                f"Unknown dataset {dataset}. ",
                "Valid values are CIFAR10, CIFAR100 and MNIST",
            )

        self.train = download_dataset(root=download_path, train=True, download=download)
        self.test = download_dataset(root=download_path, train=False, download=download)

        # TODO check: the datasets must be balanced
        self.size_class_train = len(self.train) // len(self.train.classes)
        self.size_class_test = len(self.test) // len(self.test.classes)

    def divide_dataset(self, divergence: float) -> Tuple[List[Dataset], List[Dataset]]:
        """
        divide the datasets according to the divergence.
        - 0 means to have (mostly) a uniform distribution of the classes among clients
        - 1 means to have (mostly) a single class present in each class

        Args:
            divergence:
                Float, how spread are the classes [0,1]
        """
        if not 0 <= divergence <= 1:
            raise RuntimeError(
                f"Expected divergence to be between 0 and 1, got {divergence}"
            )

        self.divergence = divergence

        # split the dataset according to the divergence
        if self.divergence == 0:
            class_clients = self.get_index(uniform=True)
        else:
            class_clients = self.get_index(b=2, uniform=False)

        # creates the local datasets
        client_datasets_train = self._create_local_dataset(
            class_clients[0], self.train.data, np.array(self.train.targets)
        )
        client_datasets_test = self._create_local_dataset(
            class_clients[1], self.train.data, np.array(self.train.targets)
        )
        self.client_datasets_train = client_datasets_train
        self.client_datasets_test = client_datasets_test

        return client_datasets_train, client_datasets_test

    def get_index(self, b: float = None, uniform: bool = True):
        """
        Calculates the indexes for each client. The indexes are then used to sample
        from the entire dataset.

        Example return:
            [
                [ # Class 0
                    [class_0_client_0],
                    [class_0_client_1],
                    ...
                ],
                [ # Class 1
                    [class_1_client_0],
                    [class_1_client_1],
                ]
                ...
            ]

        Args:
            b:
                the "beta" parameter of the Beta distributions
            uniform:
                boolean indicating whether uniform distribution

        Returns:
            class_client_mat:
                shape=(self.train.classes, self.n_client, class_elements_per_client)
        """

        # get the sorted index of the classes
        # [0,0,..0, 1,1,...,1, 2,2...9]
        sort_classes_train = np.argsort(self.train.targets)
        sort_classes_test = np.argsort(self.test.targets)

        if uniform:
            class_client_train = self.uniform_sampling(
                self.size_class_train // self.n_clients,
                sort_classes_train,
                len(self.train.classes),
                self.size_class_train
                # TODO check: not all class in training
            )
            class_client_test = self.uniform_sampling(
                self.size_class_test // self.n_clients,
                sort_classes_test,
                len(self.test.classes),
                self.size_class_test,
            )

            return class_client_train, class_client_test

    def uniform_sampling(
        self,
        class_elements_per_client: int,
        sort_label: np.array,
        n_label: int,
        size_class: int,
    ):
        """
        Each element of the returned table contains the list of index associated
        with the i_th class for the j_th client

        Args:
            class_elements_per_client:
                num. of elements we want for each class in each client

            sort_label:
                sorted array containing the labels

            n_label:
                num. of labels present ex 10 in cifar10

            size_class:
                total sample in a class

        Returns:
            matrix_class_client: Matrix (n_label, n_clients, num_sample)
        """

        start = 0
        end = self.size_class_train
        matrix_class_client = []

        for class_ in range(n_label):  # for each class

            sampling_clients = []
            for client in range(self.n_clients):  # for each client

                # TODO add: implement generator with fixed seed
                # extract randomly the selected classes from
                class_sampling = np.random.choice(
                    sort_label[start:end], class_elements_per_client, replace=False
                )
                sampling_clients.append(class_sampling)

            # next for on the next class
            start += size_class
            end += size_class
            matrix_class_client.append(np.array(sampling_clients))

        return np.array(matrix_class_client)

    def _create_local_dataset(
        self,
        class_clients_mat: np.array,
        data: np.array,
        labels: np.array,
    ):
        """
        Args:
            class_clients_mat:
                matrix shape(n_classes, n_clients, num_sample)
            data:
                it is the entire dataset (len_dataset, h, w, channel)
            labels:
                numpy array (num_labels, 1)
        """
        # local_class = (
        #     CifarLocalDataset if self.dataset.startswith("CIFAR") else MnistLocalDataset
        # )

        client_datasets = []
        for j in range(self.n_clients):
            client_indexes = class_clients_mat[:, j, :].flatten()

            unique_labels_count = np.unique(labels)

            dataset = CifarLocalDataset(
                data[client_indexes],
                labels[client_indexes],
                unique_labels_count,
                client_id=j,
            )

            client_datasets.append(dataset)

        return client_datasets


# Just here to quickly play around during development
"""
def main():
    distribute_class = DistributeDataset('CIFAR10',
                                         'data',
                                         n_clients=5)

    client_datasets = distribute_class.divide_dataset(0)
    print(client_datasets[0])
    print(client_datasets[1])

    for d, client_id in zip(client_datasets, np.arange(5)):
        print(d)
        download_path = f"data/train_dataset_client_{client_id}.pt"
        torch.save(d[0], download_path)

        download_path = f"data/test_dataset_client_{client_id}.pt"
        torch.save(d[1], download_path)

    train_dataset = torch.load(f'./data/train_dataset_client_{2}.pt')

    train_loader = DataLoader(
        train_dataset, 32, shuffle=True, num_workers=4, pin_memory=True
    )

    print(train_loader)


if __name__ == "__main__":
    main()
"""
