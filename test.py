import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import torch.nn as nn
from train import load_datasets, inspect_dataset, SentimentModel, preprocess_text, preprocess_dataset, build_word2idx_from_tokens, index_dataset, pad_sequence, pad_dataset, convert_to_tensors, train_model
import os
import yaml

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_correct = 0
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    accuracy = total_correct / len(loader.dataset)
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, f1, preds, labels


## MAIN-FUNKTION ##

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load datasets
    train_data, val_data, test_data = load_datasets()
    inspect_dataset(train_data)

    # Preprocess datasets
    train_data = preprocess_dataset(train_data)
    val_data = preprocess_dataset(val_data)
    test_data = preprocess_dataset(test_data)

    # Build word2idx and index datasets
    word2idx = build_word2idx_from_tokens(train_data['tokens'])
    print(f"Vocabulary size: {len(word2idx)}")
    import pickle
    with open("word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)

    train_data = index_dataset(train_data, word2idx)
    val_data = index_dataset(val_data, word2idx)
    test_data = index_dataset(test_data, word2idx)

    # Pad datasets
    MAX_LEN = config['data']['max_len']
    train_data = pad_dataset(train_data, MAX_LEN)
    val_data = pad_dataset(val_data, MAX_LEN)
    test_data = pad_dataset(test_data, MAX_LEN)

    # Convert to tensors
    X_train, y_train = convert_to_tensors(train_data)
    X_val, y_val = convert_to_tensors(val_data)
    X_test, y_test = convert_to_tensors(test_data)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config['data']['batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config['data']['batch_size'], shuffle=False)

    # Initialize model, loss, and optimizer
    model = SentimentModel(len(word2idx), config['model']['embedding_dim'], config['model']['hidden_dim'], config['model']['output_dim'], config['data']['max_len']).to(device)
    class_weights = torch.tensor([1.0, 3.0, 3.0, 1.0, 1.0, 1.5, 1.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Train model
    train_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=config['training']['epochs'], device=device)

    # Load the best model
    #model.load_state_dict(torch.load("best_model.pth"))
    run = wandb.init(project="MLOps", job_type="evaluate")

    artifact = run.use_artifact("sentiment_model:latest", type="model")
    artifact_dir = artifact.download()
    model_path = f"{artifact_dir}/best_model.pth"

    model.load_state_dict(torch.load(model_path))
    run.finish()

    # Evaluate on validation and test data
    val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
    test_acc, test_f1, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)

    print(f"Validation Accuracy: {val_acc * 100:.2f}%, F1-Score: {val_f1:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%, F1-Score: {test_f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pd.Categorical(test_data['Emotion']).categories)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix for Test Data')
    plt.show()

    # Plot training loss
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
