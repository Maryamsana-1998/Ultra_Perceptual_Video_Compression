#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH -o eval.out


python eval_uvg.py --original_root data/UVG/ \
                   --pred_root benchmark/unicontrol_gop8 \
                   --config configs/uni_v15_perco.yaml \
                   --ckpt ckpt/uni.ckpt \
                   --gop 8 \
                   --resolution 1080p

python eval_uvg.py --original_root data/UVG/ \
                   --pred_root benchmark/unicontrol_gop4 \
                   --config configs/uni_v15_perco.yaml \
                   --ckpt ckpt/uni.ckpt \
                   --gop 4 \
                   --resolution 1080p

python eval_uvg.py --original_root data/UVG/ \
                   --pred_root benchmark/unicontrol_gop16 \
                   --config configs/uni_v15_perco.yaml \
                   --ckpt ckpt/uni.ckpt \
                   --gop 16 \
                   --resolution 1080p