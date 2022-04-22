import pytest
import numpy as np

from datasets import DistributeUniform, DistributeDivergence
from datasets.client_distribution.utils import DatasetNameError, check_dataset_name
import math


class TestUniform:
    n_clients = 3

    def initialize_class(self, dataset, path):
        dataset = dataset
        download_path = path
        distribute_dataset = DistributeUniform(dataset, download_path, self.n_clients)
        return distribute_dataset

    @pytest.mark.parametrize(
        ["options", "sort_label", "path", "dataset"],
        [
            (
                    (
                            4,  # n_label
                            5,  # n_clients
                            2,  # client_class_e
                            3,  # size_class
                    ),
                    np.array([1, 1, 1,
                              2, 2, 2,
                              3, 3, 3,
                              4, 4, 4]),  # sort_label
                    "../data",
                    "cifar10"

            )

        ],
    )
    def test_distribute_uniform(self, options, sort_label, path, dataset):
        n_label, n_clients, client_class_e, size_class = options
        uniform_class = self.initialize_class(dataset, path)
        o = uniform_class.strategy(
            client_class_e,
            sort_label,
            n_label,
            size_class,
            n_clients)

        assert (o.shape == (n_label, n_clients, client_class_e))

    def test_rais_error(self):
        with pytest.raises(DatasetNameError):
            check_dataset_name('dataset')


class TestDivergence:
    n_clients = 3
    n_labels = 3
    size_class = 100
    smax = 1

    def initialize_class(self, dataset, path):
        dataset = dataset
        download_path = path
        distribute_dataset = DistributeDivergence(dataset, download_path, self.n_clients)
        return distribute_dataset

    @staticmethod
    def assert_list_is_close(expected, o):
        for list_clients in o:
            for c, e in zip(list_clients, expected):
                assert math.isclose(c, e, rel_tol=0.02)

    @pytest.mark.parametrize(
        ["b_params", "sets", "expected"],
        [
            ([(2, 2) for _ in range(n_labels)],
             np.linspace(0, 1, n_clients + 1),
             [0.26, 0.48, 0.26]
             ),
        ],
    )
    def test_get_percentages(self, b_params, sets, expected):
        o = DistributeDivergence.get_percentages(b_params, sets)
        o = np.array(o)
        # assert correct shape
        assert o.shape == (self.n_labels, self.n_clients)
        # sum of percentages = 1
        assert (np.sum(o, axis=1) == np.ones(self.n_labels)).all()

        self.assert_list_is_close(expected, o)

    @pytest.mark.parametrize(
        ["lp", "sc", "expected"],
        [
            ([[.2, .2, 1, .8], [.8, .8, 0, .2]],
             30,
             [0.84, 0.70]
             ),
        ],
    )
    def test_get_entropy(self, lp, sc, expected):
        lp = np.array(lp)
        e, ce, _ = DistributeDivergence.get_entropies(label_percentages=lp.transpose(),
                                                      size_class=sc)
        ce = np.array(ce)
        # check percentages
        assert ce.shape == lp.shape
        assert (np.sum(ce, axis=1) == np.ones(lp.shape[0])).all()

        # check entropies
        # one entropy for each client
        assert e.shape == (lp.shape[0],)
        for i, j in zip(e, expected):
            assert math.isclose(i, j, rel_tol=0.1)

    @pytest.mark.parametrize(
        ["div1", "div2", "params", "expected"],
        [
            (
                    0.51,
                    0.49,
                    [(2, 2) for i in range(5)],
                    [(2, 4), (2, 3), (2, 2), (3, 2), (4, 2)],

            ),
        ],
    )
    def test_update_params(self, div1, div2, params, expected):
        o1 = DistributeDivergence.update_params(params, self.smax)
        assert o1 == expected

    @pytest.mark.parametrize(
        ["params1", "expected"],
        [
            (
                    [(30, 30)] * 5,
                    ([(2, 2)] * 5)
            )
        ]
    )
    def test_update_step(self, params1, expected):
        params1 = params1
        o1 = DistributeDivergence.update_step(params1, self.smax, mean_entropy=0.1)
        assert o1 == (expected, .8)

    @pytest.mark.parametrize(
        ["div", "expected"],
        [
            (
                    [0.2, 0.4, 0.6, 0.8],
                    [
                        [[53, 25, 4], [41, 48, 41], [4, 25, 53]],
                        [[80, 25, 0], [19, 48, 19], [0, 25, 80]],
                        [[97, 25, 0], [2, 48, 2], [0, 25, 97]],
                        [[97, 25, 0], [2, 48, 2], [0, 25, 97]]

                    ]
            ),
        ],
    )
    def test_heuristic(self, div, expected):

        for i, d in enumerate(div):
            o = DistributeDivergence.heuristic(self.n_clients,
                                               self.n_labels,
                                               self.size_class,
                                               d)
            assert (o == expected[i]).all()

    @pytest.mark.parametrize(
        ["sort_label", "options", "expected", "dataset", "path"],
        [
            (
                    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    (0.4,  # div
                     4,  # size_class
                     3,  # n_clients
                     3  # n_labels
                     ),
                    [
                        [[1, 1, 1], [], []],
                        [[2], [2], [2]],
                        [[], [], [3, 3, 3]]
                    ],
                    "cifar10",
                    "../data"
            ),
        ],
    )
    def test_strategy(self, sort_label, options, expected, dataset, path):
        div, size_class, n_clients, n_labels = options
        distribute_class = self.initialize_class(dataset, path)

        o = distribute_class.strategy(-1,
                                      sort_label,
                                      n_labels,
                                      size_class,
                                      n_clients,
                                      divergence=div)

        for output, e in zip(o, expected):
            for o_v, e_v in zip(output, e):
                assert all(o_v == e_v)
