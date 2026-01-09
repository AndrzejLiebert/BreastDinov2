#!/bin/bash -l
#
# SLURM job script for DINOv2 self‑supervised training on Helios (GH200 GPUs).
#
# This script mirrors the style of the user’s Mirabel training job.  It
# assumes that a virtual environment with all required dependencies has already
# been created (see prep_env_dinov2.sh).  Edit the account name, project
# directory and dataset location before submitting.

################### SLURM SETTINGS ###################

#SBATCH --job-name=dinov2_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one process (rank) per GPU
#SBATCH --gres=gpu:1                 # request a single GH200 GPU
#SBATCH --cpus-per-task=64           # CPU cores available to PyTorch dataloaders
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=<your-grant>-gpu-gh200    # <-- CHANGE THIS
#SBATCH --output=job_outputs/dinov2_train_%j.out
#SBATCH --error=job_outputs/dinov2_train_%j.err

################### MODULES & ENV #####################

# Start from a clean module environment and load the ML‑bundle.  The bundle
# provides CUDA libraries and sets $PIP_EXTRA_INDEX_URL for downloading ARM64
# GPU wheels【869893157124862†L428-L448】.
ml ML-bundle/24.06a

################### PATHS: adjust these to your actual layout #################

# Project root (where the dinov2 package and configs reside)
PROJECT_DIR=/net/home/<username>/Work/dinov2           # <-- CHANGE if different

# Pre‑built virtualenv created by your env‑prep script
VENV_DIR=/net/scratch/<username>/dinov2_env           # <-- CHANGE if different

# Path to your DINOv2 YAML configuration (relative to PROJECT_DIR)
CONFIG_FILE=dinov2/configs/train/breast_divider.yaml

# Path to the BreastDivider dataset root.  This directory must contain
# imagesTr_batch* and labelsTr_batch* subdirectories.  Change to your actual location.
DATASET_ROOT=/net/scratch/<username>/BreastDividerDataset   # <-- CHANGE if different

# Optional: override the output directory where DINOv2 will write checkpoints
# and logs. If unset, the path defined in the YAML will be used.
# DINO_OUTPUT_DIR=/net/storage/pr3/<username>/dinov2_runs
# export DINO_OUTPUT_DIR

################### RUNTIME SETUP #####################

# Navigate to the project root
cd "$PROJECT_DIR" || exit 1

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Explicitly limit the number of OpenMP threads; ML‑bundle usually sets this, but
# we specify it here to ensure dataloaders do not oversubscribe CPU cores【869893157124862†L548-L555】.
export OMP_NUM_THREADS=1

################### LAUNCH TRAINING ###################

# Launch DINOv2 training using torchrun.  DINOv2 expects options after the
# config file to be key=value pairs that override the YAML.  We specify
# the dataset path here along with the number of data loader workers,
# optional output directory and single‑channel settings.

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m dinov2.train.train \
  --config-file "$CONFIG_FILE" \
  train.dataset_path="BreastDividerSlices:root=${DATASET_ROOT}:axis=2:mask_threshold=0.1" \
  train.num_workers="$SLURM_CPUS_PER_TASK" \
  train.output_dir="${DINO_OUTPUT_DIR:-}" \
  student.in_chans=1 \
  teacher.in_chans=1 \
  train.mri_augmentation=true \
  # Optionally override the batch size per GPU or other hyperparameters:
  # train.batch_size_per_gpu=32 \
  # train.OFFICIAL_EPOCH_LENGTH=5000

# End of script