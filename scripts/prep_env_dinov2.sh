#!/bin/bash -l
#
# SLURM job script to prepare a Python virtual environment for DINOv2 training
# on Helios (GH200 GPUs).
#
# This script closely follows the structure of the user’s `slurm_prep_env.sh`
# file.  It loads the ML‑bundle, creates a new virtual environment on
# `$SCRATCH`, removes any CPU‑only torch packages and installs a GPU build of
# PyTorch and torchvision from the Helios wheel repository.  It then
# installs additional packages required by DINOv2 such as MONAI and
# SimpleITK, and finally installs the dinov2 code in editable mode.  Edit
# the account name and paths to suit your setup before submission.

################### SLURM SETTINGS ###################

#SBATCH --job-name=dinov2_env_prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=<your-grant>-gpu-gh200    # <-- CHANGE THIS
#SBATCH --output=job_outputs/dinov2_env_prep_%j.out
#SBATCH --error=job_outputs/dinov2_env_prep_%j.err

################### MODULES & ENV #####################

# IMPORTANT: load the ML‑bundle module, which provides CUDA libraries and
# exposes `$PIP_EXTRA_INDEX_URL` pointing at the custom wheel repository.  As
# noted in the Helios documentation, you should avoid using conda on GH200
# machines【869893157124862†L428-L448】.
ml ML-bundle/24.06a

################### PATHS: adjust these to your actual layout #################

# Project root directory containing the dinov2 package and requirements.  Edit
# this path if your repository is in a different location.
PROJECT_DIR=/net/home/<username>/Work/dinov2         # <-- CHANGE if different

# Name of the virtual environment directory to create under $SCRATCH.  This
# will be created if it does not already exist.  Feel free to change the
# directory name.
VENV_NAME=dinov2_env
VENV_DIR=$SCRATCH/$VENV_NAME

################### VENV CREATION #####################

# Change to $SCRATCH to keep everything on the high‑speed parallel file system
cd "$SCRATCH" || exit 1

# Create the virtual environment if it does not already exist
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Remove any existing CPU‑only torch packages that may have been pulled in by
# the default venv creation so we can install GPU builds from the wheel repo.
pip uninstall -y torch torchvision torchaudio || true

################### INSTALL GPU TORCH #################

# Install torch and torchvision compiled for the GH200 GPUs.  Consult
# $PIP_EXTRA_INDEX_URL for available versions and adjust as needed.  The
# versions below correspond to CUDA 12.4 on ARM64; adjust if newer builds
# become available on the Helios wheel repository.
pip install --no-cache-dir torch==2.5.1+cu124.post3 torchvision==0.18.1+cu124.post3 \
  --extra-index-url "$PIP_EXTRA_INDEX_URL"

################### INSTALL DINOv2 DEPENDENCIES #############################

# Install core dependencies for DINOv2.  We avoid reinstalling torch or
# torchvision by using --no-deps where appropriate.  Additional packages
# like xformers and submitit are installed from the custom wheel repo when
# available.  Feel free to pin versions if reproducibility is critical.
pip install --no-cache-dir \
  omegaconf \
  torchmetrics==0.10.3 \
  fvcore \
  iopath \
  xformers==0.0.18 \
  submitit \
  monai[all] \
  SimpleITK \
  pyyaml

# Install the DINOv2 code in editable mode.  Using --no-deps prevents pip from
# re‑installing torch or torchvision.  Edit the path below if your
# repository location differs.
pip install --no-cache-dir --editable "$PROJECT_DIR" --no-deps

echo "Virtual environment prepared at $VENV_DIR."