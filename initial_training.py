import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ---- Samme model som før ----
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Output er stadig kun 5 klasser!
        )

    def forward(self, x):
        return self.model(x)

# ---- Opsætning ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 5
learning_rate = 0.001

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def filter_digits(dataset, allowed_digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_digits]
    return Subset(dataset, indices)

# Opgave 1: tidligere data (0–4)
train_0_4 = filter_digits(train_dataset, [0,1,2,3,4])
test_0_4 = filter_digits(test_dataset, [0,1,2,3,4])
train_loader_0_4 = DataLoader(train_0_4, batch_size=batch_size, shuffle=True)
test_loader_0_4 = DataLoader(test_0_4, batch_size=batch_size)

# Opgave 2: ny data (5–9)
train_5_9 = filter_digits(train_dataset, [5,6,7,8,9])
test_5_9 = filter_digits(test_dataset, [5,6,7,8,9])
train_loader_5_9 = DataLoader(train_5_9, batch_size=batch_size, shuffle=True)
test_loader_5_9 = DataLoader(test_5_9, batch_size=batch_size)

# ---- Initial træning på 0–4 (fra Task 1) ----
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training on digits 0–4...")
for epoch in range(epochs):
    model.train()
    for x, y in train_loader_0_4:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

# ---- Fortsat træning på 5–9 uden reset ----
print("\nContinued training on digits 5–9 (no reset, no replay)...")

acc_0_4_history = []
acc_5_9_history = []

for epoch in range(epochs):
    model.train()
    for x, y in train_loader_5_9:
        x, y = x.to(device), y.to(device)
        # Justér labels til intervallet 0–4, da modellen stadig har 5 output units
        y = y - 5
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # ---- Evaluer på gamle (0–4) og nye (5–9) efter hvert epoch ----
    def evaluate(test_loader, label_shift=0):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == (y - label_shift)).sum().item()
                total += y.size(0)
        return correct / total * 100

    acc_old = evaluate(test_loader_0_4, label_shift=0)
    acc_new = evaluate(test_loader_5_9, label_shift=5)
    acc_0_4_history.append(acc_old)
    acc_5_9_history.append(acc_new)
    print(f"Epoch {epoch+1} | Accuracy on 0–4: {acc_old:.2f}% | Accuracy on 5–9: {acc_new:.2f}%")

# ---- Plot forgetting ----
plt.plot(acc_0_4_history, label="Accuracy on 0–4 (old)")
plt.plot(acc_5_9_history, label="Accuracy on 5–9 (new)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Catastrophic Forgetting (No Replay)")
plt.legend()
plt.grid()
plt.show()
