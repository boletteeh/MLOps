import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Indlæs trænet model fra Task 1
model = torch.load("model.pt")
model.train()

# Tab og optimizer – vi inverterer gradienser manuelt senere
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Indlæs MNIST og filtrer klasse "7"
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
class_7_indices = [i for i, (x, y) in enumerate(train_dataset) if y == 7]
class_7_subset = Subset(train_dataset, class_7_indices)
unlearn_loader = DataLoader(class_7_subset, batch_size=64, shuffle=True)

# Gradient Ascent: Øg tabet for klasse 7
epochs = 3
for epoch in range(epochs):
    for data, target in unlearn_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Inverter gradienterne (Gradient Ascent)
        for param in model.parameters():
            param.grad *= -1
        
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs} – Loss: {loss.item():.4f}")

# Gem unlearned model
torch.save(model, "mnist_trained_model.pth")
