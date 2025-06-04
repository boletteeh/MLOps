import torch
from train.py import SentimentModel, load_datasets, preprocess_dataset, build_word2idx, index_dataset, pad_dataset, convert_to_tensors

# Samme parametre som under træning
EMBEDDING_DIM = 50
HIDDEN_DIM = 64
OUTPUT_DIM = 7  # eller hvor mange emotion-klasser du har
MAX_LEN = 60

# Load test data
_, _, test_data = load_datasets()
test_data = preprocess_dataset(test_data)
word2idx = build_word2idx(test_data)
test_data = index_dataset(test_data, word2idx)
test_data = pad_dataset(test_data, MAX_LEN)
X_test, y_test = convert_to_tensors(test_data)

# Load model
model = SentimentModel(vocab_size=len(word2idx), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, max_len=MAX_LEN)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Evaluer
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")
