import torch
from torch.quantization import quantize_dynamic
from train import SentimentModel  # importer din modelklasse
import nltk
nltk.download('punkt')

# Indlæs trænet model
model = SentimentModel()
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
