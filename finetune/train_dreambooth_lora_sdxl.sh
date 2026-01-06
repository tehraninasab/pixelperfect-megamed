#!/bin/bash

#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:4
#SBATCH --job-name=px
#SBATCH --output=/home/z/zahrat/workshop/pixelperfect/logs/slurm-%j.out
#SBATCH --time=24:00:00

module load python3
module load cuda
module load httpproxy

source /scratch/z/zahrat/venvs/pixelperfect-env/bin/activate

accelerate launch finetune/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="chexpert" \
  --output_dir="saved_models/sdxl_chexpert_dreambooth_lora_rank8" \
  --caption_column="prompt" \
  --instance_prompt="Chest X-ray" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=3 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --num_train_epochs=20 \
  --checkpointing_steps=100 \
  --seed="0" \
  --rank=8 \
  --num_validation_images=0 \
  --image_root_path="/scratch/z/zahrat/data/" \
  --train_csv_file="datasets/chexpert_highres/visualCheXbert_train.csv" \
  --resume_from_checkpoint="latest" \
  # --debug
  # --resume_from_checkpoint="saved_models/sdxl_chexpert_sub" \
