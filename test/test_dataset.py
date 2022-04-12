import pytest
import numpy as np
from datasets import DistributeUniform
from datasets.client_distribution.utils import DatasetNameError, check_dataset_name
from fed_torch.centralized_run import ModelError, CentralizedRun


@pytest.mark.parametrize(
    ["n_label", "n_clients", "client_class_e", "size_class", "sort_label"],
    [
        (4,  # n_label
         5,  # n_clients
         2,  # client_class_e
         3,  # size_class
         np.array([1, 1, 1,
                   2, 2, 2,
                   3, 3, 3,
                   4, 4, 4])  # sort_label
         ),
    ],
)
def test_distribute_uniform(n_label, n_clients, client_class_e, size_class, sort_label):
    strategy = DistributeUniform.strategy

    o = strategy(None,  # Self statement
                 client_class_e,
                 sort_label,
                 n_label,
                 size_class,
                 n_clients)

    assert(o.shape == (n_label, n_clients, client_class_e))


def test_rais_error():
    with pytest.raises(DatasetNameError):
        check_dataset_name('dataset')



