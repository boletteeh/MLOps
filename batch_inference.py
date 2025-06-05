import torch
from torch.utils.data import DataLoader, TensorDataset
from train import SentimentModel, preprocess_dataset, build_word2idx_from_tokens, index_dataset, pad_dataset, convert_to_tensors
import yaml

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

# Load quantized weights (for kvantiseret model skal du passe på at indlæse korrekt)
model.load_state_dict(torch.load("quantized_model.pth"))
model.eval()

# Load and preprocess test data
_, _, test_data = load_datasets()
test_data = preprocess_dataset(test_data)

# Build word2idx based on your tokens (husk at lave det ens med træningen)
token_lists = test_data['tokens'].tolist()
word2idx = build_word2idx_from_tokens(token_lists)

test_data = index_dataset(test_data, word2idx)
test_data = pad_dataset(test_data, config["data"]["max_len"])

# Convert to tensors
X_test, y_test = convert_to_tensors(test_data)

# Create DataLoader for batch inference
batch_size = 32
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Batch inference function
def batch_inference(model, dataloader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

# Run batch inference
predictions = batch_inference(model, test_loader)

print("Batch inference done. Sample predictions:", predictions[:10])
