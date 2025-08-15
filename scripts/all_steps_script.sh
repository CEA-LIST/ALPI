# all_steps_script.sh

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

# -------------------------------------------------------------------
# --- STAGE 1: RUN THE MULTI-STEP EXPERIMENT ---
# -------------------------------------------------------------------

# Run the initial step (step 0). The ratio is not needed here.
bash scripts/step_script.sh --gpu $gpu --step 0
# Run the first refinement step (step 1), passing the specified ratio.
bash scripts/step_script.sh --gpu $gpu --step 1 --ratio $ratio
# Run the second refinement step (step 2), passing the specified ratio.
bash scripts/step_script.sh --gpu $gpu --step 2 --ratio $ratio

# -------------------------------------------------------------------
# --- STAGE 2: PREPARE AND RUN THE FINAL REFINEMENT ---
# -------------------------------------------------------------------
# This stage uses the pseudo-labels generated at the end of Stage 1
# to train a final, more refined model.

bash scripts/refinement_script.sh --gpu $gpu --ratio $ratio