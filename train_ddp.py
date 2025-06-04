import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

if torch.cuda.is_available():
    print(f"✅ CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("❌ CUDA is NOT available. You're using CPU only.")

def print_rank(msg):
    if dist.is_initialized():
        print(f"[Rank {dist.get_rank()}] {msg}")
    else:
        print(f"[No DDP] {msg}")

# Importér alle nødvendige funktioner og klasser fra dit eget script
from train import (
    load_datasets, preprocess_dataset, build_word2idx, index_dataset,
    pad_dataset, convert_to_tensors, SentimentModel, train_model
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def ddp_train(rank, world_size, epochs=10, batch_size=32):
    setup(rank, world_size)

    print_rank("DDP er initialiseret og træningen starter.")

    # Datahåndtering og preprocessing (ens på alle ranks)
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

    # Brug DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Model + DDP
    model = SentimentModel(
        vocab_size=len(word2idx),
        embedding_dim=50,
        hidden_dim=64,
        output_dim=7,
        max_len=max_len
    ).to(rank)

    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Kun rank 0 skal logge til wandb
    if rank == 0:
        wandb.init(project="MLOps", job_type="ddp-train", reinit=True)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        checkpoint_path=f"best_model_rank{rank}.pth"
    )

    if rank == 0:
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(ddp_train, args=(world_size,), nprocs=world_size)

