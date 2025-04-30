#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v13
#SBATCH -o eval.out


# GPU 0
CUDA_VISIBLE_DEVICES=0 python eval_uvg.py --original_root data/UVG/ \
                   --pred_root benchmark/uvg_q1 \
                   --config configs/uni_v15_perco.yaml \
                   --ckpt ckpt/uni.ckpt \
                   --gop 8 \
                   --intra_quality 1 \
                   --resolution 1080p 

# GPU 1
CUDA_VISIBLE_DEVICES=1 python eval_uvg.py --original_root data/UVG/ \
                   --pred_root benchmark/uvg_q8 \
                   --config configs/uni_v15_perco.yaml \
                   --ckpt ckpt/uni.ckpt \
                   --gop 8 \
                   --intra_quality 8 \
                   --resolution 1080p 

# # GPU 2
# CUDA_VISIBLE_DEVICES=2 python eval_uvg.py --original_root data/UVG/ \
#                    --pred_root benchmark/unicontrol_gop16_grid15 \
#                    --config configs/uni_v15_perco.yaml \
#                    --ckpt ckpt/uni.ckpt \
#                    --gop 16 \
#                    --resolution 1080p --grid 15

wait