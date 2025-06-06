import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 5
learning_rate = 0.001

# ---- Model ----
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Kun cifre 0–4
        )

    def forward(self, x):
        return self.model(x)

# ---- Data ----
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filtrér kun cifre 0–4
def filter_digits(dataset, allowed_digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_digits]
    return Subset(dataset, indices)

train_0_4 = filter_digits(train_dataset, allowed_digits=[0, 1, 2, 3, 4])
test_0_4 = filter_digits(test_dataset, allowed_digits=[0, 1, 2, 3, 4])

train_loader = DataLoader(train_0_4, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_0_4, batch_size=batch_size)

# ---- Training ----
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ---- Evaluation ----
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

accuracy = correct / total * 100
print(f"Test Accuracy on digits 0–4: {accuracy:.2f}%")
