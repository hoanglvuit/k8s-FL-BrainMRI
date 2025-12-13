# utils.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


def get_dataloader_from_folder(folder, batch_size=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(folder, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl, len(ds.classes)

def train_one_epoch(model, dataloader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm để track tiến trình
    for step, (x, y) in enumerate(tqdm(dataloader, desc="Training"), 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        # In ra loss và accuracy sau mỗi 5 step
        if step % 5 == 0:
            current_loss = total_loss / total
            current_acc = correct / total
            print(f"Step {step}: Loss = {current_loss:.4f}, Accuracy = {current_acc:.4f}")
        
        del x, y, out, loss

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)

            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(y_true, y_pred, average="macro")  # Hoặc "micro", "weighted" tùy nhu cầu

    return avg_loss, accuracy, f1
