import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# === 1. Definer modelarkitektur ===
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# === 2. Initialiser model og load weights ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
model.load_state_dict(torch.load("mnist_trained_model.pth"))
model.train()

# === 3. Hent kun samples med klassen der skal aflæres ===
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

target_class = 7  # Den klasse vi vil "unlearn"
indices = [i for i, (x, y) in enumerate(train_dataset) if y == target_class]
subset = Subset(train_dataset, indices)
dataloader = DataLoader(subset, batch_size=64, shuffle=True)

# === 4. Unlearning med gradient ascent ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Fremad
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Baglæns (men negativt!)
        optimizer.zero_grad()
        (-loss).backward()  # Gradient ascent!
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# === 5. Gem den "aflærte" model ===
torch.save(model.state_dict(), "mnist_model_unlearned_7.pth")
print("Unlearning færdig – model gemt som 'mnist_model_unlearned_7.pth'")
