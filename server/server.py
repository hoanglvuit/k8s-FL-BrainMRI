# server.py
import flwr as fl
from flwr.server import ServerConfig
import torch
import torch.nn as nn
from model import SimpleCNN
from utils import get_dataloader_from_folder, evaluate
import os
from fedmedian import FedMedian
TESTING_DIR = os.environ.get("TESTING_DIR","/data/testing")
BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEDIAN = os.environ.get("MEDIAN","False") 


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
        loss, acc,f1_score = evaluate(model, test_loader, DEVICE, loss_fn)

        return loss, {"accuracy": acc, "f1_score":f1_score}

    return evaluate_server


if __name__ == "__main__":
    # ---- Create initial model on server ----
    model = SimpleCNN(num_classes=num_classes)
    initial_ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_ndarrays)

    # ---- Strategy with server-initialized parameters ----
    if MEDIAN == "True":
        print("Using FedMedian strategy")
        strategy = FedMedian(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(),
        initial_parameters=initial_parameters,
    )
    else:
        print("Using FedAvg strategy")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            evaluate_fn=get_evaluate_fn(),
            initial_parameters=initial_parameters,
        )

    server_config = ServerConfig(num_rounds=2)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config,
    )
