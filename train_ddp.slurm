#!/bin/bash
#SBATCH --job-name=ddp_train          # Navn på jobbet
#SBATCH --output=output_%j.log        # Logfil (j = job id)
#SBATCH --error=error_%j.log          # Fejllog
#SBATCH --partition=l4               # Partition (typisk gpu for GPU-job)
#SBATCH --gres=gpu:2                  # Antal GPU'er du ønsker (fx 2)
#SBATCH --ntasks=2                    # Antal opgaver/processor kerner
#SBATCH --time=02:00:00               # Maks køretid (timer:minutter:sekunder)
#SBATCH --mem=16G                    # Hukommelse

soruce /ect/profile

# Load miljø, fx python og CUDA (afhængigt af HPC setup)
module load python/3.10
module load cuda/12.1

# Aktivér dit virtuelle miljø, hvis du bruger det
# source ~/venv/bin/activate

# Hent data via DVS først
dvs checkout train_sent_emo.csv
dvs checkout val_sent_emo.csv

# Kør træningsscriptet med torchrun (DDP)
torchrun --nproc_per_node=2 train_ddp.py
