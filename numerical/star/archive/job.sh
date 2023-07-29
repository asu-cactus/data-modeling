#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-task 2
#SBATCH --account=m1248_g

srun -n 1 python3 train.py
