import flwr as fl
import flwr.server.strategy
import torch.nn
from flwr.common import weights_to_parameters, FitIns, EvaluateIns
from client import TorchClient
from fed_torch.Strategies.fedavgm import FedAvgM
import threading
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
from models import CNN


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

    cnn = CNN()

    for i in range(N_CLIENTS):
        x = threading.Thread(
            target=thread_function, args=([i, train_loader, test_loader, cnn])
        )
        x.start()


if __name__ == "__main__":
    main()
