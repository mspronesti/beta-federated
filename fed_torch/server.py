from typing import Dict

import flwr as fl
import hydra
import numpy as np
import torch.nn
from flwr.common import weights_to_parameters
from flwr.server.strategy import FedAvg
from torch.utils.data.dataloader import DataLoader

# from strategies import FedAvgM
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from client import TorchClient

# from datasets import DistributeDataset
from models import LeNet


def client_fn(cid, batch_size, device, n_clients):

    # Download CIFAR10
    train_dataset = CIFAR10(
        root="../../fed_torch/data/", download=True, train=True, transform=ToTensor()
    )
    test_dataset = CIFAR10(
        root="../../fed_torch/data/", download=True, train=False, transform=ToTensor()
    )

    # TODO: pass to a client only its portion of dataset and not the entire dataset

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size * 2, num_workers=4, pin_memory=True
    )

    # model
    model = LeNet()

    return TorchClient(
        client_id=cid,
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        device=device,
    )


def fit_config_fn(args: Dict):
    """Return a callable configuration function to fetch fit configurations."""

    # callback to be returned
    def fit_config(rnd):
        config = {
            "epochs": args["epochs"],
            "learning_rate": args["learning_rate"],
        }
        return config

    return fit_config


def set_all_seeds(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# TODO: move this main outside server
@hydra.main(config_path="../config/", config_name="config.yaml")
def main(cfg):
    set_all_seeds(cfg.seed)

    n_rounds = cfg.n_rounds
    fraction = cfg.fraction
    n_clients = cfg.n_clients
    epochs = cfg.epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    device = cfg.device

    model = LeNet()

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction,  # Sample 10% of available clients for the next round
        min_fit_clients=int(
            n_clients * fraction
        ),  # Minimum number of clients to be sampled for the next round
        min_available_clients=int(
            n_clients * fraction
        ),  # Minimum number of clients that need to be
        min_eval_clients=int(n_clients * fraction),
        # connected to the server before a
        # training round can start
        # server_momentum=0.9,  # TO BE CHANGED
        on_fit_config_fn=fit_config_fn({"epochs": epochs, "learning_rate": lr}),
        initial_parameters=weights_to_parameters(model.get_weights()),
    )

    clients_ids = np.arange(n_clients)

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, batch_size, device, n_clients),
        clients_ids=clients_ids,
        num_clients=n_clients,
        num_rounds=n_rounds,
        client_resources={"num_cpus": 2},
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
