
DATA_VERSION='synthesized_images_sdxl_chexpert_dreambooth_lora_rank8_without_text_interactive_epoch_3_0_-1_32_set1'

python3 scripts/eval.py \
    --chexpert_csv datasets/chexpert_highres/visualCheXbert_test.csv \
    --chexpert_root /scratch/a/amarkr1/data/ \
    --gen_csv datasets/chexpert_highres_synthetic/${DATA_VERSION}.csv \
    --gen_root /scratch/a/amarkr1/data/ \
    --out_dir metrics/${DATA_VERSION}
