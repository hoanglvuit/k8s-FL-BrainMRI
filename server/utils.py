# utils.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloader_from_folder(folder, batch_size=16, shuffle=True):
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
    for x,y in dataloader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        _,pred = out.max(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            total_loss += loss.item()*x.size(0)
            _,pred = out.max(1)
            correct += (pred==y).sum().item()
            total += x.size(0)
    return total_loss/total, correct/total
