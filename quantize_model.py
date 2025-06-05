import torch
import yaml
from torch.quantization import quantize_dynamic
from train import SentimentModel
import nltk
nltk.download('punkt')

# Indlæs konfiguration fra YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialiser model med værdier fra konfigurationen
model = SentimentModel(
    vocab_size=config["model"]["vocab_size"],
    embedding_dim=config["model"]["embedding_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    output_dim=config["model"]["output_dim"],
    max_len=config["data"]["max_len"]
)

# Indlæs trænet model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Post-training quantization
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Gem kvantiseret model
torch.save(quantized_model.state_dict(), "quantized_model.pth")
