#!/bin/bash

#SBATCH --job-name=EVAL             # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name, i.e., gpu_p
#SBATCH --gres=gpu:1            # Requests one GPU device
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=40gb                    # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=./outlog/%x.%j.out         # Standard output log
#SBATCH --error=./outlog/%x.%j.err          # Standard error log


nvidia-smi

python evaluation.py
