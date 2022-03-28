from functools import partial
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
from models import LeNet


def client_fn(client_id, model, train_loader, test_loader, device):
    return TorchClient(
        client_id=client_id,
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


@hydra.main(config_path="../config/", config_name="config.yaml")
def main(cfg):
    set_all_seeds(cfg.seed)

    # extract configs
    batch_size = cfg.batch_size
    fraction = cfg.fraction
    n_clients = cfg.n_clients
    n_rounds = cfg.n_rounds
    n_epochs = cfg.epochs
    lr = cfg.lr
    device = cfg.device

    # Download CIFAR10
    train_dataset = CIFAR10(root="data/", download=True, transform=ToTensor())
    test_dataset = CIFAR10(root="data/", train=False, transform=ToTensor())

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size * 2, num_workers=4, pin_memory=True
    )

    # model
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
        on_fit_config_fn=fit_config_fn({"epochs": n_epochs, "learning_rate": lr}),
        initial_parameters=weights_to_parameters(model.get_weights()),
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=partial(
            client_fn,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        ),
        num_clients=n_clients,
        num_rounds=n_rounds,
        client_resources={"num_cpus": 2},
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
