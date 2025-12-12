# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import SimpleCNN
from utils import get_dataloader_from_folder, train_one_epoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIENT_DATA_DIR = os.environ.get("CLIENT_DATA_DIR", "/data/client")  # mount point in pod
BATCH = 4
EPOCHS = 2  # local epochs per round

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, device):
        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):   # ← thêm config
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        for _ in range(EPOCHS):
            train_one_epoch(self.model, self.trainloader, self.device, optimizer, self.criterion)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # clients không evaluate local (server làm)
        return 0.0, len(self.trainloader.dataset), {"accuracy": 0.0}


if __name__ == "__main__":
    train_loader, num_classes = get_dataloader_from_folder(CLIENT_DATA_DIR, batch_size=BATCH, shuffle=True)
    model = SimpleCNN(num_classes=num_classes)
    client = FlowerClient(model, train_loader, DEVICE)
    fl.client.start_numpy_client(server_address=os.environ.get("SERVER_ADDRESS","server:8080"), client=client)
