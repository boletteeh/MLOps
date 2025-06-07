import deepspeed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train import (
    load_datasets, preprocess_dataset, build_word2idx, index_dataset,
    pad_dataset, convert_to_tensors, SentimentModel
)

def train_model(model, train_loader, val_loader, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            model.backward(loss)       # <-- DeepSpeed-specifik
            model.step()              # <-- DeepSpeed-specifik

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def main():
    # Data
    train_data, val_data, _ = load_datasets()
    train_data = preprocess_dataset(train_data)
    val_data = preprocess_dataset(val_data)

    word2idx = build_word2idx()
    train_data = index_dataset(train_data, word2idx)
    val_data = index_dataset(val_data, word2idx)

    max_len = 120
    train_data = pad_dataset(train_data, max_len)
    val_data = pad_dataset(val_data, max_len)

    X_train, y_train = convert_to_tensors(train_data)
    X_val, y_val = convert_to_tensors(val_data)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model og DeepSpeed
    model = SentimentModel(
        vocab_size=len(word2idx),
        embedding_dim=50,
        hidden_dim=64,
        output_dim=7,
        max_len=max_len
    )

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config="ds_config.json"
    )

    train_model(
        model=model_engine,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=10,
        device=model_engine.local_rank
    )

if __name__ == "__main__":
    main()
