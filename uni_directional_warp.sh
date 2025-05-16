#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -o experiments/uni_directional_wrap/slurm.out
#SBATCH -e experiments/uni_directional_wrap/slurm.err


# Set up directories
EXPERIMENT_DIR="experiments/uni_directional_wrap"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"

mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="configs/uni_directional_warp/local_v15.yaml"
INIT_CKPT="ckpt/uni_directional_warp/init_local.ckpt"
NUM_GPUS=8
BATCH_SIZE=1
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
    ---checkpoint-dirpath ${LOCAL_CKPT_DIR} \
    ---training-steps ${MAX_STEPS} \
    ---num-workers ${NUM_WORKERS}
