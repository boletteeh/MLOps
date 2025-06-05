import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import wandb
nltk.download('punkt')



## DATAHÅNDTERING ##

def load_datasets():
    train_data = pd.read_csv("train_sent_emo.csv")
    val_data = pd.read_csv("val_sent_emo.csv")
    test_data = pd.read_csv("test_sent_emo.csv")
    return train_data, val_data, test_data

def inspect_dataset(dataset):
    print("Dataset eksempel:")
    print(dataset.head(10))
    print("\nDatafordeling:")
    print(dataset['Emotion'].value_counts())


## TEKSTPRÆPROCESSERING ##

def preprocess_text(text):
    text = re.sub(r'[^\w\s!?.,]', '', text.lower())
    tokens = word_tokenize(text)
    return tokens

def preprocess_dataset(dataset):
    dataset['tokens'] = dataset['Utterance'].apply(preprocess_text)
    return dataset


## ORDINDEKSERING ##

def build_word2idx_from_tokens(token_lists):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for tokens in token_lists:
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1
    return word2idx

def tokens_to_indices(tokens, word2idx):
    return [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

def index_dataset(dataset, word2idx):
    dataset['indices'] = dataset['tokens'].apply(lambda tokens: tokens_to_indices(tokens, word2idx))
    return dataset
    
## PADDING OG TRIMMING ##

def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [pad_value] * (max_len - len(seq))

def pad_dataset(dataset, max_len):
    dataset['padded_indices'] = dataset['indices'].apply(lambda x: pad_sequence(x, max_len))
    return dataset


## TENSOR GENERERING ##

def convert_to_tensors(dataset):
    X = torch.tensor(dataset['padded_indices'].tolist(), dtype=torch.long)
    y = torch.tensor(pd.Categorical(dataset['Emotion']).codes, dtype=torch.long)
    return X, y


## MODELLERING ##

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 128, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.conv3 = nn.Conv1d(128, 128, 5, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(128, 128, 5, padding=1)
        self.relu4 = nn.ReLU()
        self.fc_out = nn.Linear(3840, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = x.flatten(start_dim=1)
        x = self.fc_out(x)
        return x


## TRÆNING OG EVALUERING ##

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs,
                checkpoint_path="best_model.pth"):
    best_val_loss = float('inf')
    train_losses = []
    run = wandb.init(project="MLOps", job_type="train", reinit=True)                
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")
        wandb.log({"train_loss": avg_loss, "epoch": epoch+1})

        # Validering
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch+1})

        # Gem modelcheckpoint, hvis valideringstab er forbedret
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")
 
        # W&B artifact logning
            torch.save(model.state_dict(), checkpoint_path)
            artifact = wandb.Artifact(name="sentiment_model", type="model")
            artifact.add_file(checkpoint_path)
            run.log_artifact(artifact, aliases=["best", f"epoch-{epoch+1}"])
    wandb.finish()

    return train_losses
