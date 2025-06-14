import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(data_name, batch_size=64, valid_ratio=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Fashion-MNIST 데이터셋 다운로드 (train, test)
    if data_name == "FashionMNIST":
      full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
      test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    elif data_name =="CIFAR10":
      full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
      test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Train/Validation split
    n_total = len(full_train_dataset)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_dataset, valid_dataset = random_split(full_train_dataset, [n_train, n_valid])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {n_train}, Validation samples: {n_valid}, Test samples: {len(test_dataset)}")
    return train_loader, valid_loader, test_loader

# DataLoader 실행 예시
train_loader, valid_loader, test_loader = get_data_loaders(data_name="CIFAR10",batch_size=64)
train_loader, valid_loader, test_loader = get_data_loaders(data_name="FashionMNIST",batch_size=64)


import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        out = self.linear(x)
        return out

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.output = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        out = self.output(x)
        return out
    

import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    accuracy = correct / total
    return train_loss, accuracy

def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    eval_loss = running_loss / total
    accuracy = correct / total
    return eval_loss, accuracy

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LogisticRegressionModel(input_dim=784, num_classes=10).to(device) # FashionMNIST = 28*28*1
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10
criterion = nn.CrossEntropyLoss()

print(f"FashionMNIST, LogisticRegression Model Train ...")
train_loader, val_loader, test_loader = get_data_loaders("FashionMNIST", 128)

for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)
    val_loss, val_acc = evaluate(model, val_loader, device)

     # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

# 모델 별로 test 평가 수행
# (여기서는 단순히 현재 학습된 파라미터로 평가)
val_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test Loss {val_loss:.4f}, Test Acc {test_acc:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel(input_dim=784, num_classes=10).to(device) # FashionMNIST = 28*28*1
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10
criterion = nn.CrossEntropyLoss()

print(f"FashionMNIST, MLP Model Train ...")
train_loader, val_loader, test_loader = get_data_loaders("FashionMNIST", 128)

for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)
    val_loss, val_acc = evaluate(model, val_loader, device)

     # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

# 모델 별로 test 평가 수행
# (여기서는 단순히 현재 학습된 파라미터로 평가)
val_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test Loss {val_loss:.4f}, Test Acc {test_acc:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LogisticRegressionModel(input_dim=3072, num_classes=10).to(device) # CIFAR = 32*32*3
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10
criterion = nn.CrossEntropyLoss()

print(f"CIFAR-10, LogisticRegression Model Train ...")
train_loader, val_loader, test_loader = get_data_loaders("CIFAR10", 128)

for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)
    val_loss, val_acc = evaluate(model, val_loader, device)

     # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

val_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test Loss {val_loss:.4f}, Test Acc {test_acc:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel(input_dim=3072, num_classes=10).to(device) # CIFAR = 32*32*3
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10
criterion = nn.CrossEntropyLoss()

print(f"CIFAR-10, MLP Model Train ...")
train_loader, val_loader, test_loader = get_data_loaders("CIFAR10", 128)

for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)
    val_loss, val_acc = evaluate(model, val_loader, device)

     # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

val_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test Loss {val_loss:.4f}, Test Acc {test_acc:.4f}")