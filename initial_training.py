import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import random

# Samme model som før
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.model(x)

# Opsætning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
batch_size = 64
epochs = 5
learning_rate = 0.001

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def filter_digits(dataset, allowed_digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_digits]
    return Subset(dataset, indices)

# Del datasættene op
train_0_4 = filter_digits(train_dataset, [0,1,2,3,4])
test_0_4 = filter_digits(test_dataset, [0,1,2,3,4])
train_5_9 = filter_digits(train_dataset, [5,6,7,8,9])
test_5_9 = filter_digits(test_dataset, [5,6,7,8,9])

train_loader_0_4 = DataLoader(train_0_4, batch_size=batch_size, shuffle=True)
test_loader_0_4 = DataLoader(test_0_4, batch_size=batch_size)
test_loader_5_9 = DataLoader(test_5_9, batch_size=batch_size)

# ---- Initial træning på 0–4 ----
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

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

# ---- Lav memory buffer med eksempler fra 0–4 ----
memory_size = 1000
memory_images = []
memory_labels = []

print("Storing samples in memory buffer...")
with torch.no_grad():
    for x, y in train_loader_0_4:
        for img, label in zip(x, y):
            if len(memory_images) >= memory_size:
                break
            memory_images.append(img)
            memory_labels.append(label)
        if len(memory_images) >= memory_size:
            break

memory_images = torch.stack(memory_images)
memory_labels = torch.tensor(memory_labels)
memory_dataset = TensorDataset(memory_images, memory_labels)

# ---- Træning på 5–9 med experience replay ----
train_loader_5_9 = DataLoader(train_5_9, batch_size=batch_size, shuffle=True)
acc_0_4_history = []
acc_5_9_history = []

print("\nTraining on digits 5–9 + replaying memory from 0–4...")

for epoch in range(epochs):
    model.train()
    for x_new, y_new in train_loader_5_9:
        x_new, y_new = x_new.to(device), y_new.to(device)
        y_new = y_new - 5  # justér til 0–4 output-interval

        # Hent replay samples
        replay_indices = random.sample(range(len(memory_dataset)), k=min(32, len(memory_dataset)))
        x_replay = memory_images[replay_indices].to(device)
        y_replay = memory_labels[replay_indices].to(device)

        # Kombiner replay + nye data
        x_combined = torch.cat([x_new, x_replay], dim=0)
        y_combined = torch.cat([y_new, y_replay], dim=0)

        optimizer.zero_grad()
        outputs = model(x_combined)
        loss = criterion(outputs, y_combined)
        loss.backward()
        optimizer.step()

    # Evaluer
    def evaluate(loader, label_shift=0):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == (y - label_shift)).sum().item()
                total += y.size(0)
        return 100 * correct / total

    acc_old = evaluate(test_loader_0_4, label_shift=0)
    acc_new = evaluate(test_loader_5_9, label_shift=5)
    acc_0_4_history.append(acc_old)
    acc_5_9_history.append(acc_new)
    print(f"Epoch {epoch+1} | Accuracy on 0–4: {acc_old:.2f}% | Accuracy on 5–9: {acc_new:.2f}%")

# ---- Plot ----
plt.plot(acc_0_4_history, label="Accuracy on 0–4 (old)")
plt.plot(acc_5_9_history, label="Accuracy on 5–9 (new)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Experience Replay: Mitigating Forgetting")
plt.legend()
plt.grid()
plt.show()

