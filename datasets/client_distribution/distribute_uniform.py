import hydra

from .distribute_dataset import DistributeDataset
import numpy as np


class DistributeUniform(DistributeDataset):

    def strategy(
            self,
            client_class_e: int,
            sort_label: np.array,
            n_label: int,
            size_class: int,
            n_clients: int,
            *args
    ):
        """
        Each element of the returned table contains the list of index associated
        with the i_th class for the j_th client

        Args:
            client_class_e:
                num. of elements we want for each class in each client
            sort_label:
                sorted array containing the labels
            n_label:
                num. of labels present ex 10 in cifar10
            size_class:
                total sample in a class
            n_clients:
                the number of clients
        Returns:
            matrix_class_client: Matrix (n_label, n_clients, num_sample)
        """
        matrix_class_client = [
            np.random.choice(  # TODO add: implement generator with fixed seed
                sort_label[l * size_class: (l + 1) * size_class],
                client_class_e,
                replace=False
            )
            for l in range(n_label)
            for _ in range(n_clients)
        ]

        return np.reshape(matrix_class_client, (n_label, n_clients, client_class_e))
