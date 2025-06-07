# inference.py

import torch
import pickle
import pandas as pd
from train import (
    SentimentModel,
    preprocess_text,
    build_word2idx_from_tokens,
    pad_sequence
)

def run_single_inference():
    # Indlæs testdata
    test_data = pd.read_csv("test_sent_emo.csv")

    # Forbered input
    sample_text = test_data['Utterance'].iloc[0]
    tokens = preprocess_text(sample_text)

    # Byg ord-til-indeks mapping baseret på testdata
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    
    indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    padded = pad_sequence(indices, max_len=69)
    input_tensor = torch.tensor([padded], dtype=torch.long)

    # Indlæs model
    model = SentimentModel(vocab_size=len(word2idx), embedding_dim=50, hidden_dim=50, output_dim=7, max_len=69)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Udfør inferens
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predikteret klasse: {predicted_class}")
