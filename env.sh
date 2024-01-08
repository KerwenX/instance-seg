# System config
export OMP_NUM_THREADS=1
export NUM_WORKERS=4
export SEED=2021

export DATA=..

# NOTE: Change the data config based on your detup!
# JSON files
export DATA_DIR=$DATA/Data/Dataset
# Resized images with intrinsics and poses
export IMAGE_ROOT=$DATA/Data/Images
# Depths and instances rendered over images
export RENDERING_ROOT=$DATA/Data/Rendering
# Scan2CAD Full Annotations
export FULL_ANNOT=$DATA/Data/full_annotations.json

# Model configurations
export RETRIEVAL_MODE=resnet_resnet+image+comp
export E2E=1
export NOC_WEIGHTS=1

# Train and test behavior
export EVAL_ONLY=1
export CHECKPOINT=$DATA/Models/model_best.pth  # "none"
export RESUME=0  # This means from last checkpoint
export OUTPUT_DIR=output
