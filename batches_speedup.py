import torch
from torch.utils.data import DataLoader, TensorDataset
from train import SentimentModel, preprocess_dataset, load_datasets, build_word2idx_from_tokens, index_dataset, pad_dataset, convert_to_tensors
import yaml
import time
import matplotlib.pyplot as plt

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load and initialize quantized model
model = SentimentModel(
    vocab_size=config["model"]["vocab_size"],
    embedding_dim=config["model"]["embedding_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    output_dim=config["model"]["output_dim"],
    max_len=config["data"]["max_len"]
)

# Load quantized model state_dict (hvis du har gemt state_dict)
# Hvis du har gemt hele modellen, brug i stedet model = torch.load(...)
model = torch.load("quantized_model.pth")
model.eval()

device = torch.device("cpu")
model.to(device)

# Load and preprocess test data
_, _, test_data = load_datasets()
test_data = preprocess_dataset(test_data)

# Build word2idx
token_lists = test_data['tokens'].tolist()
word2idx = build_word2idx_from_tokens(token_lists)

test_data = index_dataset(test_data, word2idx)
test_data = pad_dataset(test_data, config["data"]["max_len"])

# Convert to tensors
X_test, y_test = convert_to_tensors(test_data)

# Flyt data til device (kan også gøres batchvis)
X_test = X_test.to(device)
y_test = y_test.to(device)

batch_sizes = [1, 8, 16, 32, 64]
throughputs = []
latencies = []

for bs in batch_sizes:
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=bs, shuffle=False)
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            total_samples += inputs.size(0)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = total_samples / total_time
    latency = total_time / total_samples

    throughputs.append(throughput)
    latencies.append(latency)

    print(f"Batch size: {bs}, Samples: {total_samples}, Time: {total_time:.4f}s, Throughput: {throughput:.2f} samples/s, Latency: {latency*1000:.2f} ms/sample")

# Plot throughput og latency
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(batch_sizes, throughputs, marker='o')
plt.title("Throughput vs Batch size")
plt.xlabel("Batch size")
plt.ylabel("Throughput (samples/second)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(batch_sizes, latencies, marker='o', color='orange')
plt.title("Latency vs Batch size")
plt.xlabel("Batch size")
plt.ylabel("Latency (seconds per sample)")
plt.grid(True)

plt.tight_layout()
plt.savefig("batch_inference_performance.png")
plt.show()
