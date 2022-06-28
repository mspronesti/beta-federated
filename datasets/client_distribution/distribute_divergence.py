import math

from typing import List, Tuple
from .distribute_dataset import DistributeDataset
import numpy as np
from scipy.stats import beta, entropy


class HeuristicError(Exception):
    def __init__(self):
        self.message = "The heuristics was not able to find the correct split"
        super().__init__(self.message)


class DistributeDivergence(DistributeDataset):
    # Todo: define max elements
    # Todo: define min elements
    def strategy(
        self,
        client_class_e: int,
        sort_label: List[int],
        n_label: int,
        size_class: int,
        n_clients: int,
        divergence: float,
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
            divergence:
                b parameter used in the beta distribution

        Returns:
            matrix_class_client: Matrix (n_label, n_clients, ...)
            the last dimension is not fixed since
            the num of elements for the same class is different for different clients
        """
        elements = self.heuristic(n_clients, n_label, size_class, divergence)
        # row labels, column client
        elements = np.transpose(elements)

        matrix_class_client = [
            np.random.choice(  # TODO add: implement generator with fixed seed
                sort_label[l * size_class : (l + 1) * size_class], client, replace=False
            )
            for l, clients in enumerate(elements)
            for client in clients
        ]
        return np.reshape(matrix_class_client, (n_label, n_clients))

    @classmethod
    def heuristic(
        cls, n_clients: int, n_label: int, size_class: int, divergence: float
    ) -> List[List]:
        """
        It returns per each client, the number of elements per each class.
        The split of the dataset is computed according to the divergence
        ( entropy = 1 - divergence ).

        This heuristic exploits the shape of the beta distribution which is controlled
        by the parameters a and b. After the split, it is computed the entropy
        for each client w.r.t the number of elements in each class.
        The mean of the entropies is compared with (1 - divergence)

        Args:
            n_clients: num of clients
            n_label: num of labels, 10 for cifar10
            size_class: max num of elements in a class
            divergence: value between 0 and 1 (min, max respectively) .

        Returns:
            elements: (n_clients, n_labels),
                      for each client the num of element for each class
        """
        # Todo: this happens only with small number of clients
        #  entropy get stuck around 0.33 with n_clients = 3
        if divergence > 0.6:
            divergence = 0.6

        sets = np.linspace(0, 1, n_clients + 1)
        # default b creates gaussians
        params = np.array([(2.0, 2.0) for _ in range(n_label)])
        step = 1
        while True:
            label_percentages = cls.get_percentages(params, sets)

            entropies, _, elements = cls.get_entropies(label_percentages, size_class)

            mean_entropy = float(np.mean(entropies))

            # Stop condition
            if math.isclose(mean_entropy, 1 - divergence, rel_tol=0.1):
                return elements.astype("int16")

            # if already passed there is no need to go until entropy=0
            if mean_entropy < 1 - divergence:
                params, step = cls.update_params([(30, 30)] * 5, step)

            # Update params
            params = cls.update_params(params, step)

            # update steps
            params, step = cls.update_step(params, step, mean_entropy)

            # Not able to find the correct split
            if step == 0:
                raise HeuristicError()

    @staticmethod
    def update_step(
        params: List[Tuple], step: int, mean_entropy: float
    ) -> Tuple[List[Tuple], int]:
        """
        When an iteration is completed:
            - reset to default the params
            - decrease the step of 0.2
        An iteration is completed when:
            - a or b is greater than 30
            - entropy is equal to 0

        Args:
            params: array of tuple [(a,b), ...]
            step: step of the current iteration, default 1
            mean_entropy: entropy of the current iteration

        Returns:
            params: update to default params
            step: update step decreased of 0.2
        """
        max_b = params[0][1]
        max_a = params[-1][0]
        # divergence > 0.5
        if (max_b >= 100 or max_a >= 100) or mean_entropy == 0:
            params = [(2, 2)] * len(params)
            step -= 0.2

        return params, step

    @staticmethod
    def update_params(params: List[Tuple], step: int) -> List[Tuple]:
        """
        update the params according to the step size.
        first half of the array updates the "b" param i.e. skew distribution to the left
        second half of the array updates the "a" param i.e. skew distribution to the right
        Args:
            params: array of tuple [(a,b),...]
            step:  current step

        Returns:
            params: update according to step
        """
        l = len(params) // 2

        left = [(a, b + step * (l - i)) for i, (a, b) in enumerate(params[:l])]

        right = [(a + step * i, b) for i, (a, b) in enumerate(params[l:])]

        params = left + right
        return params

    @staticmethod
    def get_entropies(
        label_percentages: List[List], size_class: int
    ) -> Tuple[np.array, np.array, np.array]:
        """
        given the percentages for each client and the total element in a class,
        calculate the entropy associated for each client

        Args:
            label_percentages: (n_label, n_clients) for each distribution,
                               the area in each set
            size_class: tot element in a class, 1000 in cifar 10

        Returns:
            entropies: the normalized entropies associated with each client
            probs: for each client, the probability to find each class
            class_elements: for each client, the num of element for each class

        """
        # get the actual number of elements per class
        class_elements = np.array([np.array(l) * size_class for l in label_percentages])
        # row client, column labels
        class_elements = class_elements.transpose()
        # total elements assigned for each client
        total_elem = np.sum(class_elements, axis=1)[..., None]  # column vector
        # for each client, probabilities to find the classes
        probs = np.divide(class_elements, total_elem)

        entropies = entropy(probs, axis=1, base=2)
        # normalize in order to obtain an entropy between 0 and 1
        num_classes = probs.shape[1]
        entropies = entropies / np.log2(num_classes)

        return entropies, probs, class_elements

    @staticmethod
    def get_percentages(params: np.array, sets: np.array) -> List[List[float]]:
        """
        calculate the integral of the beta distributions in each set
        a set is the area expressed by two adjacent points on the x-axis
        Args:
            params: params distribution [(a,b), ...]
            sets: equidistant points on the x-axis [0, 0.33, ..., 1],
                  difference between adjacent points represents 1 client

        Returns:
            label_percentages: (n_label, n_clients) for each distribution,
                               the area in each set
        """
        label_percentages = []
        for (a, b) in params:
            # calculate the area from 0 to x
            beta_cdf = beta.cdf(sets, a, b)
            # calculate the area from x to x+1
            beta_areas = [
                beta_cdf[i + 1] - beta_cdf[i] for i in range(len(beta_cdf) - 1)
            ]
            label_percentages.append(beta_areas)
        return label_percentages
