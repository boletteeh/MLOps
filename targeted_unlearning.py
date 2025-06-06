import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# === 1. Definer modelarkitektur der matcher din gemte model ===
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

# === 2. Initialiser og load weights ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
model.load_state_dict(torch.load("mnist_trained_model.pth"))
model.train()

# === 3. Hent kun samples med den klasse der skal aflæres ===
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

target_class = 7
indices = [i for i, (x, y) in enumerate(train_dataset) if y == target_class]
subset = Subset(train_dataset, indices)
dataloader = DataLoader(subset, batch_size=64, shuffle=True)

# === 4. Gradient ascent ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        (-loss).backward()  # gradient ascent
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# === 5. Gem den aflærte model ===
torch.save(model.state_dict(), "mnist_model_unlearned_7.pth")
print("✅ Unlearning færdig – model gemt som 'mnist_model_unlearned_7.pth'")


# --- Evalueringsfunktion ---
def evaluate_forgetting(model, dataloader, forget_class=7):
    correct_forget = 0
    total_forget = 0
    
    correct_others = 0
    total_others = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Separate forget_class og andre klasser
            mask_forget = (labels == forget_class)
            mask_others = (labels != forget_class)
            
            # Eval for forget_class
            if mask_forget.sum() > 0:
                correct_forget += (predicted[mask_forget] == labels[mask_forget]).sum().item()
                total_forget += mask_forget.sum().item()
            
            # Eval for andre klasser
            if mask_others.sum() > 0:
                correct_others += (predicted[mask_others] == labels[mask_others]).sum().item()
                total_others += mask_others.sum().item()
    
    acc_forget = correct_forget / total_forget if total_forget > 0 else 0
    acc_others = correct_others / total_others if total_others > 0 else 0
    
    return acc_forget, acc_others

# --- Kør evaluering ---
# === Tilføj test_loader til evaluering ===
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluer på testdata (indeholder både klasse 7 og alle andre)
acc_forget, acc_others = evaluate_forgetting(model, test_loader, forget_class=7)

print(f"Accuracy på aflærte klasse 7: {acc_forget*100:.2f}%")
print(f"Accuracy på øvrige klasser: {acc_others*100:.2f}%")
