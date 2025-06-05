import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from train import (
    load_datasets, preprocess_dataset, build_word2idx, index_dataset,
    pad_dataset, convert_to_tensors, SentimentModel, train_model
)

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    # Dataforberedelse
    train_data, val_data, _ = load_datasets()
    train_data = preprocess_dataset(train_data)
    val_data = preprocess_dataset(val_data)

    word2idx = build_word2idx()
    train_data = index_dataset(train_data, word2idx)
    val_data = index_dataset(val_data, word2idx)

    max_len = 120
    train_data = pad_dataset(train_data, max_len)
    val_data = pad_dataset(val_data, max_len)

    X_train, y_train = convert_to_tensors(train_data)
    X_val, y_val = convert_to_tensors(val_data)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

    # Model og træning
    model = SentimentModel(
        vocab_size=len(word2idx),
        embedding_dim=50,
        hidden_dim=64,
        output_dim=7,
        max_len=max_len
    ).to(device)

    model = DDP(model, device_ids=[device])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if rank == 0:
        wandb.init(project="MLOps", job_type="ddp-train", reinit=True)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=10,
        checkpoint_path=f"best_model_rank{rank}.pth",
        device=device
    )

    if rank == 0:
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    main()
