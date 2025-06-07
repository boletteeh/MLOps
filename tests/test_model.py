import sys
import os
import yaml
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from train import SentimentModel  # Juster efter dit projekt

# Indlæs config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

VOCAB_SIZE = config["model"]["vocab_size"]
EMBEDDING_DIM = config["model"]["embedding_dim"]
HIDDEN_DIM = config["model"]["hidden_dim"]
OUTPUT_DIM = config["model"]["output_dim"]
MAX_LEN = config["data"]["max_len"]

# Opret en testmodel
@pytest.fixture
def model():
    return SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, MAX_LEN)

def test_model_initialization(model):
    assert isinstance(model, SentimentModel), "Modelen initialiseres ikke korrekt"

def test_model_forward(model):
    sample_input = torch.randint(0, VOCAB_SIZE, (2, MAX_LEN), dtype=torch.long)
    output = model(sample_input)
    assert output.shape == (2, OUTPUT_DIM), "Modelens output har forkert form"
