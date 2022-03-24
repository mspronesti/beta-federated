import argparse
from subprocess import Popen


my_parser = argparse.ArgumentParser(description='Server')

my_parser.add_argument('--number_of_clients',
                       type=int,
                       default=30,
                       help='number of clients launched')

args = my_parser.parse_args()

number_of_clients = args.number_of_clients

Popen(['python3', './server.py']) # something long running


for i in range(number_of_clients):
    print(f"Starting client {i}")
    Popen(['python3', './client.py'])
    # ODO: Launch client