#!/bin/bash

# Define paths to your data
TRAIN_CSV="/home/amarkr/pixelperfect/datasets/chexpert/chexpert_train.csv"
VAL_CSV="/home/amarkr/pixelperfect/datasets/chexpert/chexpert_val.csv"
TEST_CSV="/home/amarkr/pixelperfect/datasets/chexpert/chexpert_test.csv"
REAL_ROOT="/home/jupyter"
SYNTHETIC_ROOT="."

# Base directory for experiments
BASE_DIR="image_size_experiments"
mkdir -p ${BASE_DIR}

# Model parameters
MODEL="resnet50"
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=50
LEARNING_RATE=1e-4

# Different image sizes to test
IMAGE_SIZES=(64 128 256 512)

# Run training for each image size
for size in "${IMAGE_SIZES[@]}"; do
    echo "Starting training with image size: ${size}x${size}"
    
    # Create output directory
    OUTPUT_DIR="${BASE_DIR}/${MODEL}_size_${size}"
    mkdir -p ${OUTPUT_DIR}
    
    # Run the training script
    python train_chexpert.py \
        --train_csv ${TRAIN_CSV} \
        --validation_csv ${VAL_CSV} \
        --test_csv ${TEST_CSV} \
        --real_root ${REAL_ROOT} \
        --synthetic_root ${SYNTHETIC_ROOT} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --num_epochs ${NUM_EPOCHS} \
        --num_workers ${NUM_WORKERS} \
        --base_classifier ${MODEL} \
        --image_size ${size} \
        --pretrained
    
    echo "Completed training with image size: ${size}x${size}"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "--------------------------------------------------------"
done

# Create a summary file
echo "Creating summary of results..."
SUMMARY_FILE="${BASE_DIR}/image_size_comparison.csv"
echo "image_size,accuracy,auroc,precision,recall,f1" > ${SUMMARY_FILE}

for size in "${IMAGE_SIZES[@]}"; do
    METRICS_FILE="${BASE_DIR}/${MODEL}_size_${size}/test_metrics.csv"
    
    if [ -f "${METRICS_FILE}" ]; then
        # Extract mean metrics from the file (assuming the mean is in a specific row)
        MEAN_METRICS=$(grep "Mean" ${METRICS_FILE} | cut -d',' -f2-)
        echo "${size},${MEAN_METRICS}" >> ${SUMMARY_FILE}
    else
        echo "${size},N/A,N/A,N/A,N/A,N/A" >> ${SUMMARY_FILE}
    fi
done

echo "Summary saved to: ${SUMMARY_FILE}"
echo "All image size experiments completed!"