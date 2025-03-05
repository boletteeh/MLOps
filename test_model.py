import torch
import pytest
from train import SentimentModel  # Juster importen efter din filstruktur

# Opsæt testparametre
VOCAB_SIZE = 1000
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
OUTPUT_DIM = 3
MAX_LEN = 30

# Opret en testmodel
@pytest.fixture
def model():
    return SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, MAX_LEN)

# Test modelinitialisering
def test_model_initialization(model):
    assert isinstance(model, SentimentModel), "Modelen initialiseres ikke korrekt"

# Test forward-metoden
def test_model_forward(model):
    sample_input = torch.randint(0, VOCAB_SIZE, (2, MAX_LEN))  # Simulerer batch med 2 sekvenser
    output = model(sample_input)
    assert output.shape == (2, OUTPUT_DIM), "Modelens output har forkert form"

# Test om modellen kan lave en fremadpassage uden fejl
def test_model_forward_pass_no_error(model):
    sample_input = torch.randint(0, VOCAB_SIZE, (4, MAX_LEN))
    try:
        output = model(sample_input)
    except Exception as e:
        pytest.fail(f"Modelen fejlede under forward pass: {e}")