#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --nodes=1               # antal noder
#SBATCH --ntasks-per-node=4     # antal processer (typisk = antal GPU'er pr node)
#SBATCH --gres=gpu:4            # antal GPU'er pr node
#SBATCH --cpus-per-task=4       # CPU kerner pr task (kan justeres)
#SBATCH --time=02:00:00         # max køretid
#SBATCH --output=ddp_out_%j.log # logfil

module load cuda/11.7   # load relevant CUDA-modul (tjek AI Lab modulet)
module load anaconda3   # eller andet miljøsystem
#source activate dit_miljø  # aktivér dit conda/miljø med PyTorch installeret

torchrun --nproc_per_node=4 your_script.py
