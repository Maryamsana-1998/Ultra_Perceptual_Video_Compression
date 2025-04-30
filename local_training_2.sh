#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -o experiments/exp_video_bi/slurm.out
#SBATCH -e experiments/exp_video_bi/slurm.err


# Set up directories
EXPERIMENT_DIR="experiments/exp_vimeo_bi"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"

mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="configs/local_v15_r1_op_r2.yaml"
INIT_CKPT="experiments/exp_video_bi/local_ckpt/local-best-checkpoint.ckpt"
NUM_GPUS=8
BATCH_SIZE=2
NUM_WORKERS=16
MAX_STEPS=100000


# Copy config file to experiment directory
cp ${CONFIG_PATH} ${EXPERIMENT_DIR}/local_v15.yaml
echo "Config file copied to ${EXPERIMENT_DIR}/local_v15.yaml"

# Create a JSON file of training hyperparameters
HYPERPARAM_FILE="${EXPERIMENT_DIR}/hyperparams.json"

cat <<EOF > ${HYPERPARAM_FILE}
{
    "num_gpus": ${NUM_GPUS},
    "batch_size": ${BATCH_SIZE},
    "num_workers": ${NUM_WORKERS},
    "max_steps": ${MAX_STEPS},
    "config": "${CONFIG_PATH}",
    "init_ckpt": "${INIT_CKPT}",
    "loss":"baseline+lpips normfix without logvar",
    "data": "100"
}
EOF

echo "Hyperparameters JSON saved at ${HYPERPARAM_FILE}"

# Run Training
python src/train/train.py \
    --config-path ${CONFIG_PATH} \
    ---resume-path ${INIT_CKPT} \
    ---gpus ${NUM_GPUS} \
    ---batch-size ${BATCH_SIZE} \
    ---logdir ${LOGS_DIR} \
    --checkpoint-dirpath ${LOCAL_CKPT_DIR} \
    ---training-steps ${MAX_STEPS} \
    ---num-workers ${NUM_WORKERS}

# After training, prepare uni weights
LOCAL_BEST="${LOCAL_CKPT_DIR}/local-best-checkpoint.ckpt"
UNI_CKPT="${EXPERIMENT_DIR}/uni.ckpt"
UNI_CONFIG="configs/uni_v15_bi.yaml"

python utils/prepare_weights.py integrate \
       ${LOCAL_BEST} ckpt/init_global_temp_bi.ckpt  \
       ${UNI_CONFIG} ${UNI_CKPT} 

echo "Unified weights prepared and stored at ${UNI_CKPT}."
echo "Experiment finished successfully."

python eval_uvg.py --original_root data/UVG \
                   --pred_root ${PRED_DIR}  \
                   --config ${UNI_CONFIG} \
                   --ckpt ${UNI_CKPT} \
                   --gop 8 \
                   --intra_quality 4 \

