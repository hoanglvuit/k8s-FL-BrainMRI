import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import SimpleCNN
from utils import get_dataloader_from_folder, train_one_epoch

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 2

BASE_DATASET_DIR = "../dataset"
CLIENTS_DIR = os.path.join(BASE_DATASET_DIR, "Clients")
TEST_DIR = os.path.join(BASE_DATASET_DIR, "Testing")

# ================= DATA =================
def get_centralized_trainloader(batch_size):
    client_datasets = []
    num_classes = None

    for client_name in sorted(os.listdir(CLIENTS_DIR)):
        client_path = os.path.join(CLIENTS_DIR, client_name)
        if os.path.isdir(client_path):
            dl, n_cls = get_dataloader_from_folder(
                client_path,
                batch_size=batch_size,
                shuffle=True
            )
            client_datasets.append(dl.dataset)
            num_classes = n_cls

    centralized_dataset = ConcatDataset(client_datasets)
    train_loader = DataLoader(
        centralized_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, num_classes


def get_testloader(batch_size):
    test_loader, _ = get_dataloader_from_folder(
        TEST_DIR,
        batch_size=batch_size,
        shuffle=False
    )
    return test_loader


# ================= EVALUATION =================
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


# ================= MAIN =================
def main():
    train_loader, num_classes = get_centralized_trainloader(BATCH_SIZE)
    test_loader = get_testloader(BATCH_SIZE)

    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print("🚀 Centralized Training Started")
    for epoch in range(EPOCHS):
        train_one_epoch(
            model,
            train_loader,
            DEVICE,
            optimizer,
            criterion
        )
        print(f"Epoch [{epoch+1}/{EPOCHS}] finished")

    acc, f1 = evaluate(model, test_loader, DEVICE)
    print("\n📊 Evaluation on Testing set")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()
