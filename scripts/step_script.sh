# step_script.sh


#!/bin/bash

# -----------------
# --- SETUP & CONFIGURATION ---
# -----------------
set -e

default_gpu=0
default_step=0
default_ratio=100

# -----------------
# --- ARGUMENT PARSING ---
# -----------------

function show_usage {
    echo "Usage: $0 [--step CURRENT_REFINEMENT_STEP] [--ratio RATIO_FAKE_TO_REAL] [--gpu GPU_TO_USE]"
}


while [[ $# -gt 0 ]]; do
    key="$1" 
    case $key in
        --gpu)
            gpu="$2" 
            shift
            ;;
        --step)
            step="$2"
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


# Set the final values for gpu, step, and ratio.
gpu=${gpu:-$default_gpu}
step=${step:-$default_step}
ratio=${ratio:-$default_ratio}

# Display the configuration that the script will run with.
echo "GPU: $gpu"
echo "Step: $step"
echo "Ratio: $ratio"

# -----------------
# --- DIRECTORY & FILE PREPARATION ---
# -----------------
Category="carpedcyc" # Defines the dataset/class category
CFG_ARG="cfgs/det_sample.yaml" # Path to the model configuration file

# Conditionally set the output directory name based on the step number.
if (($step > 0)); then
    OUTPUT_DIR=output/"$Category"_step_0"$step"_ratio_0"$ratio"
else
    OUTPUT_DIR=output/"$Category"_step_0"$step"
fi
echo "Save output directory is $OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

# Copy the configuration file into the output directory for reproducibility.
cp "$CFG_ARG" "$OUTPUT_DIR"

# -----------------
# --- DATA GENERATION ---
# -----------------
PREV=$(( $step - 1 ))
if (($step > 0)); then
    # --- Refinement Steps (step > 0) ---
    # These steps generate new data using pseudo-labels from the PREVIOUS step.
    # Generate new training instances.
    python kitti/generate_instances_insert.py --step $PREV --ratio $ratio
    # Create the final pickle file with the new data mix.
    python kitti/frustum_pseudolabeling.py --gen_train --step $PREV --ratio $ratio
else
    # --- Initial Step (step == 0) ---
    python ./kitti/frustum_extractor.py --gen_train
fi

# -----------------
# --- MODEL TRAINING & EVALUATION ---
# -----------------

# Training
CUDA_VISIBLE_DEVICES=$gpu python train/train_net_det.py --cfg "$CFG_ARG" --step $step --ratio $ratio --weak $CAT_ARG OUTPUT_DIR "$OUTPUT_DIR"

# Evaluation
CUDA_VISIBLE_DEVICES=$gpu python train/test_net_det.py --cfg "$CFG_ARG" OUTPUT_DIR "$OUTPUT_DIR/_last_val" TEST.WEIGHTS "$OUTPUT_DIR/model_0050.pth" &

# Pseudolabeling
CUDA_VISIBLE_DEVICES=$gpu python train/gen_pseudo.py --cfg "$CFG_ARG" OUTPUT_DIR "$OUTPUT_DIR/_last_train" TEST.WEIGHTS "$OUTPUT_DIR/model_0050.pth" 

wait