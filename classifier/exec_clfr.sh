#!/bin/bash

# Define common parameters
TRAIN_CSV="/home/amarkr/pixelperfect/datasets/chexpert_highres/visualCheXbert_train.csv"
VAL_CSV="/home/amarkr/pixelperfect/datasets/chexpert_highres/visualCheXbert_val.csv"
TEST_CSV="/home/amarkr/pixelperfect/datasets/chexpert_highres/visualCheXbert_test.csv"
REAL_ROOT="/home/amarkr/data"
SYNTHETIC_ROOT="."
BASE_DIR="results_clfr_highres"
NUM_WORKERS=4
LEARNING_RATE=1e-4
NUM_EPOCHS=10
NUM_GPUS=4
MEMORY_THRESHOLD=1000  # MB - consider GPU free if memory usage < this

# Create base directory
mkdir -p ${BASE_DIR}

# Array to track background process PIDs and their GPU assignments
declare -a PIDS=()
declare -A PID_GPU_MAP=()

# Function to get free GPUs
get_free_gpus() {
    local free_gpus=()
    
    if ! command -v nvidia-smi &> /dev/null; then
        for ((i=0; i<NUM_GPUS; i++)); do
            free_gpus+=($i)
        done
        echo "${free_gpus[@]}"
        return
    fi
    
    # Get GPU memory usage
    local gpu_memory_usage
    gpu_memory_usage=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    
    while IFS=', ' read -r gpu_id memory_used; do
        # Remove any whitespace
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        memory_used=$(echo "$memory_used" | tr -d ' ')
        
        # Check if GPU is free (memory usage below threshold)
        if [ "$memory_used" -lt "$MEMORY_THRESHOLD" ]; then
            # Also check if we don't have a running job on this GPU
            local gpu_occupied=false
            for pid in "${PIDS[@]}"; do
                if [ "${PID_GPU_MAP[$pid]}" = "$gpu_id" ] && kill -0 "$pid" 2>/dev/null; then
                    gpu_occupied=true
                    break
                fi
            done
            
            if [ "$gpu_occupied" = false ]; then
                free_gpus+=($gpu_id)
            fi
        fi
    done <<< "$gpu_memory_usage"
    
    echo "${free_gpus[@]}"
}

# Function to wait for a free GPU
wait_for_free_gpu() {
    while true; do
        # Clean up completed processes
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                unset PID_GPU_MAP[$pid]
            fi
        done
        PIDS=("${new_pids[@]}")
        
        # Check for free GPUs
        local free_gpus_array=($(get_free_gpus))
        if [ ${#free_gpus_array[@]} -gt 0 ]; then
            echo "${free_gpus_array[0]}"
            return
        fi
        
        sleep 30
    done
}

# Function to run training
run_training() {
    local image_size=$1
    local model=$2
    local use_pretrained=$3
    local batch_size=$4
    local gpu_id=$5
    
    # Create output directory name
    local pretrained_str=""
    if [ "$use_pretrained" = "true" ]; then
        pretrained_str="_pretrained"
    fi
    
    local output_dir="${BASE_DIR}/${model}_${image_size}${pretrained_str}"
    
    # Check if already completed
    if [ -d "$output_dir" ] && [ -n "$(find "$output_dir" -name "*.pth" -print -quit 2>/dev/null)" ]; then
        echo "Skipping ${model}_${image_size}${pretrained_str} - already completed"
        return
    fi
    

    
    # Build command
    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train_chexpert.py \
        --train_csv ${TRAIN_CSV} \
        --validation_csv ${VAL_CSV} \
        --test_csv ${TEST_CSV} \
        --real_root ${REAL_ROOT} \
        --synthetic_root ${SYNTHETIC_ROOT} \
        --output_dir ${output_dir} \
        --batch_size ${batch_size} \
        --learning_rate ${LEARNING_RATE} \
        --num_epochs ${NUM_EPOCHS} \
        --num_workers ${NUM_WORKERS} \
        --base_classifier ${model} \
        --image_size ${image_size}"
    
    # Add pretrained flag if needed
    if [ "$use_pretrained" = "true" ]; then
        cmd="${cmd} --pretrained"
    fi
    
    # Create output directory and run
    mkdir -p "${output_dir}"
    eval ${cmd} > "${output_dir}/training.log" 2>&1 &
    
    # Store PID and GPU mapping
    local pid=$!
    PIDS+=($pid)
    PID_GPU_MAP[$pid]=$gpu_id
}

# Function to show GPU status
show_gpu_status() {
    echo "=== GPU Status ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    else
        echo "nvidia-smi not available"
    fi
    echo "=================="
}

# Main execution
echo "Starting training jobs..."

# Define all configurations
declare -a CONFIGS=()

for size in 64 128 256 512 1024; do
    for model in "resnet50" "efficientnet-b0" "densenet121"; do
        for pretrained in true false; do
            # Set batch size based on image size
            case $size in
                64|128) BATCH_SIZE=1024 ;;
                256) BATCH_SIZE=256 ;;
                512) BATCH_SIZE=64 ;;
                1024) BATCH_SIZE=16 ;;
                *) BATCH_SIZE=32 ;;
            esac
            
            # Adjust for DenseNet
            if [ "$model" = "densenet121" ]; then
                BATCH_SIZE=$((BATCH_SIZE / 4))
                if [ "$BATCH_SIZE" -lt 1 ]; then
                    BATCH_SIZE=1
                fi
            fi
            
            CONFIGS+=("$size,$model,$pretrained,$BATCH_SIZE")
        done
    done
done

# Process all configurations
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r size model pretrained batch_size <<< "$config"
    
    # Wait for a free GPU
    gpu_id=$(wait_for_free_gpu)
    
    # Run training on the free GPU
    run_training $size $model $pretrained $batch_size $gpu_id
    
    sleep 5  # Small delay between launches to allow GPU memory allocation
done

# Wait for all remaining jobs to complete
while [ ${#PIDS[@]} -gt 0 ]; do
    local new_pids=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        else
            unset PID_GPU_MAP[$pid]
        fi
    done
    PIDS=("${new_pids[@]}")
    
    if [ ${#PIDS[@]} -gt 0 ]; then
        sleep 30
    fi
done

echo "All training jobs completed!"

# Generate summary report
echo "Training Results Summary" > ${BASE_DIR}/summary.txt
echo "=======================" >> ${BASE_DIR}/summary.txt
date >> ${BASE_DIR}/summary.txt
echo "" >> ${BASE_DIR}/summary.txt

# Count completed experiments
completed=0
total=0
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r size model pretrained batch_size <<< "$config"
    total=$((total + 1))
    
    pretrained_str=""
    if [ "$pretrained" = "true" ]; then
        pretrained_str="_pretrained"
    fi
    
    output_dir="${BASE_DIR}/${model}_${size}${pretrained_str}"
    if [ -d "$output_dir" ] && [ -n "$(find "$output_dir" -name "*.pth" -print -quit 2>/dev/null)" ]; then
        completed=$((completed + 1))
        echo "✓ ${model}_${size}${pretrained_str}" >> ${BASE_DIR}/summary.txt
    else
        echo "✗ ${model}_${size}${pretrained_str}" >> ${BASE_DIR}/summary.txt
    fi
done

echo "" >> ${BASE_DIR}/summary.txt
echo "Completed: $completed/$total experiments" >> ${BASE_DIR}/summary.txt