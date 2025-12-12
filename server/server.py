# server.py
import flwr as fl
from flwr.server import ServerConfig
import torch
import torch.nn as nn
from model import SimpleCNN
from utils import get_dataloader_from_folder, evaluate
import os

TESTING_DIR = os.environ.get("TESTING_DIR","/data/testing")
BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load testing dataloader to evaluate global model
test_loader, num_classes = get_dataloader_from_folder(TESTING_DIR, batch_size=BATCH, shuffle=False)

def get_evaluate_fn():
    def evaluate_server(server_round, parameters, config):
        # Build model
        model = SimpleCNN(num_classes=num_classes)
        state_dict = model.state_dict()

        # Convert parameters list → tensors → state_dict
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict)

        # Evaluate
        loss_fn = nn.CrossEntropyLoss()
        loss, acc = evaluate(model, test_loader, DEVICE, loss_fn)

        return loss, {"accuracy": acc}

    return evaluate_server


if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # all clients
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn()
    )
    server_config = ServerConfig(num_rounds=10)
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=server_config)
