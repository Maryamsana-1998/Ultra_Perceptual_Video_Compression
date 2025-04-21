#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -o experiments/vimeo_temp/slurm.out
#SBATCH -e experiments/vimeo_temp/slurm.err

# echo "Running on $(hostname)"
# nvidia-smi

# Set up directories
EXPERIMENT_DIR="experiments/vimeo_temp"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"


mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="configs/local_v15_r1_op_r2_d.yaml"
INIT_CKPT="ckpt/init_local_video.ckpt"
NUM_GPUS=2
BATCH_SIZE=1
NUM_WORKERS=8
MAX_EPOCHS=3


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
    "max_epochs": ${MAX_EPOCHS},
    "config": "${CONFIG_PATH}",
    "init_ckpt": "${INIT_CKPT}",
    "loss":"baseline+lpips normfix without logvar",
    "data": "40"
}
EOF

echo "Hyperparameters JSON saved at ${HYPERPARAM_FILE}"

# Run Training
python src/train/train_sub.py \
    --config-path ${CONFIG_PATH} \
    ---resume-path ${INIT_CKPT} \
    ---gpus ${NUM_GPUS} \
    ---batch-size ${BATCH_SIZE} \
    ---logdir ${LOGS_DIR} \
    --checkpoint-dirpath ${LOCAL_CKPT_DIR} \
    ---max-epochs ${MAX_EPOCHS} \
    ---num-workers ${NUM_WORKERS}

# # After training, prepare uni weights
# LOCAL_BEST="${LOCAL_CKPT_DIR}/local-best-checkpoint.ckpt"
# UNI_CKPT="${EXPERIMENT_DIR}/uni.ckpt"
# UNI_CONFIG="configs/uni_v15.yaml"

# python utils/prepare_weights.py integrate \
#        ${LOCAL_BEST} ckpt/init_global_video.ckpt  \
#        ${UNI_CONFIG} ${UNI_CKPT} 

# echo "Unified weights prepared and stored at ${UNI_CKPT}."
# echo "Experiment finished successfully."

# python eval_uvg.py --original_root data/UVG_test_data \
#                    --pred_root ${PRED_DIR}  \
#                    --config ${UNI_CONFIG} \
#                    --ckpt ${UNI_CKPT}

