import time
import torch
import yaml
from torch.quantization import quantize_dynamic
from train import SentimentModel, load_datasets, preprocess_dataset, build_word2idx_from_tokens, index_dataset, pad_dataset, convert_to_tensors, evaluate_model
import nltk
nltk.download('punkt')

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize model
model = SentimentModel(
    vocab_size=config["model"]["vocab_size"],
    embedding_dim=config["model"]["embedding_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    output_dim=config["model"]["output_dim"],
    max_len=config["data"]["max_len"]
)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Load and preprocess test data
_, _, test_data = load_datasets()
test_data = preprocess_dataset(test_data)
word2idx = build_word2idx()
test_data = index_dataset(test_data, word2idx)
test_data = pad_dataset(test_data, config["data"]["max_len"])
X_test, y_test = convert_to_tensors(test_data)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32)

# Function to measure inference time
def measure_inference_time(model, loader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in loader:
            _ = model(inputs)
    end_time = time.time()
    return end_time - start_time

# Evaluate accuracy function is imported from train.py as evaluate_model
criterion = torch.nn.CrossEntropyLoss()

# Evaluate original model accuracy and inference time
orig_acc, orig_f1, _, _ = evaluate_model(model, test_loader, criterion)
orig_time = measure_inference_time(model, test_loader)

print(f"Original model - Accuracy: {orig_acc:.4f}, F1: {orig_f1:.4f}, Inference time: {orig_time:.2f} s")

# Quantize model
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Evaluate quantized model accuracy and inference time
quantized_acc, quantized_f1, _, _ = evaluate_model(quantized_model, test_loader, criterion)
quantized_time = measure_inference_time(quantized_model, test_loader)

print(f"Quantized model - Accuracy: {quantized_acc:.4f}, F1: {quantized_f1:.4f}, Inference time: {quantized_time:.2f} s")

# Save quantized model
torch.save(quantized_model.state_dict(), "quantized_model.pth")
