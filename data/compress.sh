#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH -o compress.out


python compress_downsample_flows.py