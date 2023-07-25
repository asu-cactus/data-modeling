#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --mem=200G
#SBATCH --cpus-per-task 32
#SBATCH --gpus 4
#SBATCH --account=m1248_g

srun -n 1 python train.py
