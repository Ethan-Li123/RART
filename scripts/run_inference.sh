#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# Single GPU: use CUDA_VISIBLE_DEVICES=0 and --num_processes 1; remove --multi_gpu.
export CUDA_VISIBLE_DEVICES=0,1

python -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --gpu_ids all \
    --main_process_port 29501 \
    scripts/inference_temporal.py \
    --checkpoint_dir weights/Wan2.2-TI2V-5B \
    --motion_ckpt outputs/checkpoint-200000 \
    --dismo_repo_path DisMo \
    --dismo_ckpt_path weights/motion_extractor_large.pt \
    --dino_ckpt_path weights/dinov2_vitl14_reg.pth \
    --inference_csv data/RART_test_data/inference.csv \
    --output_dir outputs/inference \
    --csv_start 0 \
    --csv_limit -1 \
    --t5_cpu_offload \
    "$@"
