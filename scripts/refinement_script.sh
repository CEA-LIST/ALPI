# refinement_script.sh

#!/bin/bash

# -----------------
# --- SETUP & CONFIGURATION ---
# -----------------
set -e

# Define default values for script parameters.
default_gpu=0
default_ratio=100

# -----------------
# --- ARGUMENT PARSING ---
# -----------------

function show_usage {
    echo "Usage: $0 [--ratio RATIO_FAKE_TO_REAL] [--gpu GPU_TO_USE]"
}

while [[ $# -gt 0 ]]; do
    key="$1" 
    
    case $key in
        --gpu)
            gpu="$2"
            shift 
            ;;
        --ratio)
            ratio="$2"
            shift
            ;;
        --help)
            show_usage
            exit 0 
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1 
            ;;
    esac
    shift
done


# Set the final values for gpu and ratio.
gpu=${gpu:-$default_gpu}
ratio=${ratio:-$default_ratio}

# Display the configuration that the script will run with.
echo "GPU: $gpu"
echo "Ratio: $ratio"


# Hardcode the output directory from the last step of the previous stage.
Category="carpedcyc"
CFG_ARG="cfgs/refine_sample.yaml"

OUTPUT_DIR=output/"$Category"_step_02_ratio_0"$ratio"
source_dir_val="$OUTPUT_DIR/_last_val/val_nms/result/data"
source_dir_train="$OUTPUT_DIR/_last_train/val_nms/result/data"
# Define the target directory where the new pseudo-labels will be stored.
target_dir="data/kitti/training/label_2_pseudo"
if [ ! -d "$target_dir" ]; then
    mkdir -p "$target_dir"
else
    rm -f "$target_dir"/*
fi

echo "Pseudo-labels loaded from $OUTPUT_DIR"
# Copy all generated pseudo-label files from the source directories into the single target directory.
cp "$source_dir_val"/* "$target_dir"
cp "$source_dir_train"/* "$target_dir"

echo "Pseudo-labels copied to $target_dir"

# Define a new output directory for the final refinement training run.
OUTPUT_DIR=output/"$Category"_train_refine
echo "Save output directory is $OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

# Copy the config and data preparation script for reproducibility.
cp "$CFG_ARG" "$OUTPUT_DIR"
cp ./kitti/prepare_data_refine.py "$OUTPUT_DIR"

# Run the data preparation script for the refinement stage.
python kitti/prepare_data_refine.py --gen_train --gen_val

# Train the final refinement model using the new data.
CUDA_VISIBLE_DEVICES=$gpu python train/train_net_det.py --cfg "$CFG_ARG" OUTPUT_DIR "$OUTPUT_DIR"
# Evaluate the final model on the validation set. The '&' runs this in the background.
CUDA_VISIBLE_DEVICES=$gpu python train/test_net_det.py --cfg "$CFG_ARG" OUTPUT_DIR "$OUTPUT_DIR/_last_val" TEST.WEIGHTS "$OUTPUT_DIR/model_last.pth" &
# Generate a new set of pseudo-labels with the final model.
CUDA_VISIBLE_DEVICES=$gpu python train/gen_pseudo.py --cfg "$CFG_ARG" OUTPUT_DIR "$OUTPUT_DIR/_last_train" TEST.WEIGHTS "$OUTPUT_DIR/model_last.pth"

wait