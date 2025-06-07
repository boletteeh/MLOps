import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Modelarkitektur
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 output classes
        )

    def forward(self, x):
        return self.model(x)

# Enheder og hyperparametre
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 5
learning_rate = 0.001

# MNIST data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model, optimizer, loss
model = Classifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Træning
print("Training on all MNIST digits (0–9)...")
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done.")

# Evaluering
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            for i in range(len(y)):
                label = y[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    acc_total = 100 * correct / total
    acc_per_class = [100 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    return acc_total, acc_per_class

# Resultater
overall_acc, class_acc = evaluate(model, test_loader)
print(f"\n✅ Total accuracy: {overall_acc:.2f}%")
for i, acc in enumerate(class_acc):
    print(f"Class {i} accuracy: {acc:.2f}%")

# Gem modellen til brug i Task 2
torch.save(model.state_dict(), "mnist_trained_model.pth")
