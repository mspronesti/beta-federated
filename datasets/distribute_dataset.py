import numpy as np
import torchvision
from cifar_local_dataset import CifarLocalDataset

# from mnist_local_dataset import MnistLocalDataset


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
        client_datasets:
            the clients LocalDataset after the datasets split
        sample_per_class:
            num of sample per each class inside the datasets
    """

    def __init__(
        self,
        dataset,
        download_path,
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
        dataset = dataset.upper()

        self.dataset = dataset
        self.divergence = None
        self.client_datasets = None

        assert n_clients > 0
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

        # TODO check: the datasets must be balance
        self.sample_per_class = len(self.train) // len(self.train.classes)

    def divide_dataset(self, divergence):
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
        client_datasets = self._create_local_dataset(
            self.n_clients, class_clients, self.train.data, np.array(self.train.targets)
        )
        self.client_datasets = client_datasets

        return client_datasets

    def get_index(self, b=None, uniform=True):
        """
        Args:
            b: b parameter of the beta distributions
            uniform: Boolean indicating whether uniform distribution

        Returns:
            class_client_mat: (self.train.classes, self.n_client, element_per_class)
        """

        # get the sorted index of the classes
        # [0,0,..0, 1,1,...,1, 2,2...9]
        sort_classes_i = np.argsort(self.train.targets)

        if uniform:
            # number of class elements per client
            element_per_class = self.sample_per_class // self.n_clients

            class_client_mat = self.uniform_sampling(
                self.n_clients,
                element_per_class,
                self.sample_per_class,
                sort_classes_i,
                len(self.train.classes)
                # TODO check: not all class in training
            )

            expected = (len(self.train.classes), self.n_clients, element_per_class)
            if class_client_mat.shape != expected:
                raise RuntimeError(
                    f"Expected shape {expected}, got {class_client_mat.shape}"
                )

            return class_client_mat

    def uniform_sampling(self, n_clients, num_sample, tot_class, sort_label, n_label):
        """
        Each element of the returned table contains the list of index associated
        with the i_th class for the j_th client

        Args:
            n_clients:
                num of clients
            num_sample:
                num. of sample we want for each class and client
            tot_class:
                tot. num of element in the classes
            sort_label:
                sorted array containing the list to scan
            n_label:
                num. of labels present ex 10 in cifar10

        Returns:
            matrix_class_client: Matrix (n_label, n_clients, num_sample)
        """

        start = 0
        end = tot_class
        matrix_class_client = []

        for class_ in range(n_label):  # for each class
            sampling_clients = []
            for client in range(n_clients):  # for each client

                # TODO add: implement generator with fixed seed
                # extract randomly the selected classes from
                class_sampling = np.random.choice(
                    sort_label[start:end],
                    num_sample,
                )
                sampling_clients.append(class_sampling)

            # next for on the next class
            start += tot_class
            end += tot_class
            matrix_class_client.append(np.array(sampling_clients))

        return np.array(matrix_class_client)

    def _create_local_dataset(
        self,
        n_clients,
        class_clients_mat,
        data,
        labels,
    ):
        """
        Args:
            n_clients:
                number of clients
            class_clients_mat:
                matrix (n_classes, n_clients, num_sample)
            data:
                numpy array (batch_size, h, w, channel)
            labels:
                numpy array (num_labels, 1)
        """
        # local_class = (
        #     CifarLocalDataset if self.dataset.startswith("CIFAR") else MnistLocalDataset
        # )

        client_datasets = []
        for j in range(n_clients):
            client_indexes = class_clients_mat[:, j].flatten()

            unique_labels_count = np.unique(labels)

            dataset = CifarLocalDataset(
                data[client_indexes],
                labels[client_indexes],
                unique_labels_count,
                client_id=j,
            )

            client_datasets.append(dataset)

        return client_datasets
