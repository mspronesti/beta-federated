import flwr as fl

def main():
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for the next round
        min_fit_clients=5,  # Minimum number of clients to be sampled for the next round
        min_available_clients=20,  # Minimum number of clients that need to be connected to the server before a training round can start
    )

    # Start Flower server for ten rounds of federated learning
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 10},
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
