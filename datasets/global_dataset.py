import numpy as np
import torchvision

from datasets import CifarLocalDataset


class DivideDataset:
    """
    :attribute train: train datasets
    :attribute test: test datasets
    :attribute n_client: number of clients
    :attribute divergence: how spread are the classes [0,1]
    :attribute client_range: array containing the x-range on beta distr.
    :attribute client_datasets: the clients LocalDataset after the datasets split
    :attribute sample_per_class: num of sample per each class inside the datasets
    """

    def __init__(self,
                 dataset,
                 download_path,
                 download=False,
                 n_client=1,
                 ):
        """
        :param dataset: String, containing the datasets ['CIFAR10', 'CIFAR100', 'MNIST']
        :param download_path: String, path for the datasets
        :param download: Boolean, to download the datasets or not


        """
        self.divergence = None

        assert (n_client > 0)
        self.n_client = n_client
        self.client_range = np.linspace(0, 1, n_client + 1)

        if dataset == 'CIFAR10':
            self.train = torchvision.datasets.CIFAR10(root=download_path,
                                                      train=True,
                                                      download=download)
            self.test = torchvision.datasets.CIFAR10(root=download_path,
                                                     train=False,
                                                     download=download)

        elif dataset == 'CIFAR100':
            self.train = torchvision.datasets.CIFAR100(root=download_path,
                                                       train=True,
                                                       download=download)
            self.test = torchvision.datasets.CIFAR100(root=download_path,
                                                      train=False,
                                                      download=download)

        elif dataset == 'MNIST':
            self.train = torchvision.datasets.MNIST(root=download_path,
                                                    train=True,
                                                    download=download)
            self.test = torchvision.datasets.MNIST(root=download_path,
                                                   train=False,
                                                   download=download)

        else:
            assert False

        # TODO check: the datasets must be balance
        self.sample_per_class = len(self.train) // len(self.train.classes)

    def divide_dataset(self, divergence):
        """
        divide the datasets according to the divergence.
        - 0 means to have (mostly) a uniform distribution of the classes among clients
        - 1 means to have (mostly) a single class present in each class

        :param divergence: Float, how spread are the classes [0,1]
        """

        assert (0 <= divergence <= 1)
        self.divergence = divergence

        if self.divergence == 0:
            class_clients = self.get_index(uniform=True)
        else:
            class_clients = self.get_index(a=2, b=2, uniform=False)

        datasets = self._create_local_dataset(self.n_client,
                                              class_clients,
                                              self.train.data,
                                              np.array(self.train.targets))
        return datasets

    def get_index(self,
                  a=None,
                  b=None,
                  uniform=True
                  ):
        """
        :param a:
        :param b:
        :param uniform:

        """
        # get the sorted index of the classes
        # [0,0,..0, 1,1,...,1, 2,2...9]
        sort_classes_i = np.argsort(self.train.targets)

        if uniform:
            # number of class elements per client
            element_per_class = self.sample_per_class // self.n_client

            class_client_mat = self._uniform_sampling(self.n_client,
                                                      element_per_class,
                                                      self.sample_per_class,
                                                      sort_classes_i,
                                                      len(self.train.classes)
                                                      # TODO check: not all class in training
                                                      )

            assert (class_client_mat.shape[0] == len(self.train.classes))
            assert (class_client_mat.shape[1] == self.n_client)
            assert (class_client_mat.shape[2] == element_per_class)

            return class_client_mat

    def _uniform_sampling(
            self,
            n_client,
            num_sample,
            tot_class,
            sort_label,
            n_label
    ):
        """
        Each element of the returned table contains the list of index associated
        with the i_th class for the j_th client

        :param n_client: num of clients
        :param num_sample: num. of sample we want for each class and client
        :param tot_class: tot. num of element in the classes
        :param sort_label: sorted array containing the list to scan
        :param n_label: num. of labels present ex 10 in cifar10

        :return matrix_class_client: Matrix (n_label, n_clients, num_sample)
        """

        start = 0
        end = tot_class
        matrix_class_client = []
        print('tot_class = ', n_label)
        for class_ in range(n_label):  # for each class
            sampling_clients = []
            for client in range(n_client):  # for each client

                # TODO add: implement generator with fixed seed
                # extract randomly the selected classes from
                class_sampling = np.random.choice(sort_label[start:end],
                                                  num_sample,
                                                  )
                sampling_clients.append(class_sampling)

            # next for on the next class
            start += tot_class
            end += tot_class
            matrix_class_client.append(np.array(sampling_clients))

        return np.array(matrix_class_client)

    def _create_local_dataset(self,
                              n_client,
                              class_clients_mat,
                              data,
                              labels,
                              ):
        """
        :param n_client: number of clients
        :param class_clients_mat: matrix (n_classes, n_client, num_sample)
        :param data: numpy array (batch_size, h, w, channel)
        :param labels: numpy array (num_labels, )

        """
        datasets = []
        for j in range(n_client):
            client_indexes = class_clients_mat[:, j].flatten()

            print('client_indexes => ', client_indexes.shape)
            unique_labels_count = np.unique(labels)

            dataset = CifarLocalDataset(data[client_indexes],
                                        labels[client_indexes],
                                        unique_labels_count,
                                        client_id=j)

            datasets.append(dataset)

        return datasets
