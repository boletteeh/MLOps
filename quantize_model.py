import torch
import yaml
from torch.quantization import quantize_dynamic
from train import SentimentModel  # importer din modelklasse
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Load konfiguration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialiser model med konfigurationsparametre
model = SentimentModel(
    vocab_size=config["vocab_size"],
    embedding_dim=config["embedding_dim"],
    hidden_dim=config["hidden_dim"],
    output_dim=config["output_dim"],
    max_len=config["max_len"]
)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Anvend post-training quantization
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # eller flere lagtyper hvis relevant
    dtype=torch.qint8
)

# Gem den kvantiserede model
torch.save(quantized_model.state_dict(), "quantized_model.pth")
