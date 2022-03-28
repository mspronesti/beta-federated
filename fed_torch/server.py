from typing import Dict

import flwr as fl
import numpy as np
import torch.nn
from client import TorchClient

# from strategies import FedAvgM
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
from models import LeNet
from functools import partial
from flwr.common import weights_to_parameters
from flwr.server.strategy import FedAvg

"""
def thread_function(i, train_loader, test_loader, model: torch.nn.Module):
    print(f"Started client {i}")
    # model equal cnn is temporary, just to test
    client = TorchClient(
        client_id=i,
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
    )

    fit_ins = FitIns(
        parameters=weights_to_parameters(model.get_weights()),
        config={"epochs": 1, "learning_rate": 0.1},
    )

    fit_res = client.fit(ins=fit_ins)

    evaluate_ins = EvaluateIns(parameters=fit_res.parameters, config={})
    client.evaluate(ins=evaluate_ins)


def start_server(strategy: flwr.server.strategy.Strategy) -> None:
    # Start Flower server for ten rounds of federated learning
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 10},
        strategy=strategy,
    )


def main():
    # Define strategy
    strategy = FedAvgM(
        fraction_fit=0.1,  # Sample 10% of available clients for the next round
        min_fit_clients=5,  # Minimum number of clients to be sampled for the next round
        min_available_clients=20,  # Minimum number of clients that need to be
        # connected to the server before a
        # training round can start
        server_momentum=0.9,  # TO BE CHANGED
    )

    x = threading.Thread(target=start_server, args=([strategy]))
    x.start()

    N_CLIENTS = 10

    # Download CIFAR10
    train_dataset = CIFAR10(root="data/", download=True, transform=ToTensor())
    test_dataset = CIFAR10(root="data/", train=False, transform=ToTensor())

    batch_size = 16

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size * 2, num_workers=4, pin_memory=True
    )

    for i in range(N_CLIENTS):
        x = threading.Thread(
            target=thread_function, args=([i, train_loader, test_loader, CNN()])
        )
        x.start()


if __name__ == "__main__":
    main()
"""
######################################
# PLEASE NOTICE:
# all this stuff is going to
# be parametrized  !
# This is just to test if it does work
#######################################

# define num_clients
N_CLIENTS = 10
# define num rounds
N_ROUNDS = 2
# define the size of each opt batch
BATCH_SIZE = 16
# fraction
FRACTION = 0.1
# global seed
SEED = 2022


def client_fn(client_id, model, train_loader, test_loader):
    return TorchClient(
        client_id=client_id,
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
    )


def fit_config_fn(args: Dict):
    """Return a callable configuration function to fetch fit configurations."""

    # callback to be returned
    def fit_config(rnd):  # non so a che serva, mi costringe a passarglielo
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


def main():
    set_all_seeds(SEED)

    # dataset
    # Download CIFAR10
    train_dataset = CIFAR10(root="data/", download=True, transform=ToTensor())
    test_dataset = CIFAR10(root="data/", train=False, transform=ToTensor())

    train_loader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, BATCH_SIZE * 2, num_workers=4, pin_memory=True
    )

    # model
    model = LeNet()

    # Define strategy
    strategy = FedAvg(
        fraction_fit=FRACTION,  # Sample 10% of available clients for the next round
        min_fit_clients=int(
            N_CLIENTS * FRACTION
        ),  # Minimum number of clients to be sampled for the next round
        min_available_clients=int(
            N_CLIENTS * FRACTION
        ),  # Minimum number of clients that need to be
        min_eval_clients=int(N_CLIENTS * FRACTION),
        # connected to the server before a
        # training round can start
        # server_momentum=0.9,  # TO BE CHANGED
        on_fit_config_fn=fit_config_fn({"epochs": 10, "learning_rate": 0.001}),
        initial_parameters=weights_to_parameters(model.get_weights()),
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=partial(
            client_fn, model=model, train_loader=train_loader, test_loader=test_loader
        ),
        num_clients=N_CLIENTS,
        num_rounds=N_ROUNDS,
        client_resources={"num_cpus": 2},
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
