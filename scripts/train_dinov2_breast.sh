#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=<your-grant>-gpu-gh200
#SBATCH --output=dinov2_breast_%j.out
#SBATCH --error=dinov2_breast_%j.err

##############################################################################
# Slurm submission script for self‑supervised pretraining of DINOv2 on the
# BreastDivider dataset. This script targets the Helios GPU partition (GH‑200)
# using a single node and GPU. Adjust the ``--gres`` and ``--gpus`` arguments
# to scale to multiple GPUs if desired and ensure your grant supports the
# requested resources. Replace ``<your-grant>`` with your actual PLGrid grant
# name suffixed with ``-gpu-gh200`` as required by the Helios accounting model.
##############################################################################

set -euo pipefail

# Load the ML bundle to set up CUDA and a custom wheel repository for ARM
ml ML-bundle/24.06a

# Limit the number of threads spawned by multiprocessing libraries. When the
# ML‑bundle is loaded this is set automatically, but redefining here ensures
# that subprocesses inherit the correct value.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

##############################################################################
# Prepare Python environment
# A virtual environment is recommended over conda on Helios. The environment is
# created in the job's scratch directory. Subsequent jobs can reuse the same
# environment by caching it on $SCRATCH or your project directory.
##############################################################################

WORKDIR=${SCRATCH:-$HOME}/dinov2_breast_job_${SLURM_JOB_ID}
mkdir -p "$WORKDIR"
cd "$WORKDIR"

VENV_DIR=$WORKDIR/venv
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install PyTorch and relevant libraries from the Helios wheel repository. Versions
# listed below are examples; use ``ls -1 $PIP_EXTRA_INDEX_URL`` to see available
# wheels and pick the appropriate versions for your job.
pip install --no-cache-dir torch==2.5.1+cu124.post3 torchvision==0.18.1+cu124.post3 --extra-index-url "$PIP_EXTRA_INDEX_URL"

# Install MONAI, SimpleITK and other dependencies from PyPI. These packages are
# pure Python or provide ARM wheels on Helios. If a package is missing, you may
# need to build it from source.
pip install --no-cache-dir monai[all] SimpleITK pyyaml scipy

# Install the DINOv2 codebase in editable mode. Adjust the path below if your
# checked‑out repository is elsewhere. Using ``--no-deps`` prevents pip from
# attempting to reinstall torch or torchvision.
REPO_PATH=$WORKDIR/dinov2
if [ ! -d "$REPO_PATH" ]; then
    # copy repository from your home or project directory. Replace this with the
    # actual location of the modified dinov2 repository.
    cp -r /path/to/your/dinov2 "$REPO_PATH"
fi
pip install --no-cache-dir --editable "$REPO_PATH" --no-deps

##############################################################################
# Dataset location
# Set BREAST_DIVIDER_ROOT to the directory where you unpacked the BreastDivider
# dataset. The directory must contain ``imagesTr_batch*`` and ``labelsTr_batch*``
# subdirectories as described in the dataset card. Update this path accordingly.
##############################################################################

export BREAST_DIVIDER_ROOT=/path/to/BreastDividerDataset

if [ ! -d "$BREAST_DIVIDER_ROOT" ]; then
    echo "Error: BREAST_DIVIDER_ROOT ($BREAST_DIVIDER_ROOT) does not exist." >&2
    exit 1
fi

##############################################################################
# Launch training
# We use the standalone launch for DINOv2 because this script targets a single
# GPU. For multi‑GPU training you can replace the invocation with ``torchrun``
# and set ``--gpus-per-node`` accordingly. The configuration file defines most
# hyper‑parameters; only the dataset path and output directory are overridden
# here via the command line.
##############################################################################

OUTPUT_DIR=$WORKDIR/output
mkdir -p "$OUTPUT_DIR"

python -m dinov2.train.train \
  --config-file "$REPO_PATH/dinov2/configs/train/breast_divider.yaml" \
  train.dataset_path="BreastDividerSlices:root=${BREAST_DIVIDER_ROOT}:axis=2:mask_threshold=0.1" \
  train.output_dir="$OUTPUT_DIR" \
  student.in_chans=1 teacher.in_chans=1 \
  train.mri_augmentation=true

echo "Training completed. Check $OUTPUT_DIR for logs and checkpoints."