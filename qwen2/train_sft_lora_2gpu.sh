#! /usr/bin/env bash

set -ex

LR=2e-4
RUN_NAME=hotel_qwen25_sft_lora_2gpu
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"
DEEPSPEED_CONFIG="deepspeed_zero2.json"
USE_DEEPSPEED=${USE_DEEPSPEED:-0}

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

DEEPSPEED_ARGS=()
if [[ "$USE_DEEPSPEED" == "1" ]]; then
    DEEPSPEED_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
    echo "[Info] Using DeepSpeed ZeRO2 mode"
else
    echo "[Info] Using stable DDP mode (DeepSpeed disabled)"
fi

torchrun --nproc_per_node=2 --master_port=29511 finetune.py \
    --do_train \
    --do_eval \
    --train_file ../data/train.jsonl \
    --validation_file ../data/dev.jsonl \
    --prompt_column context \
    --response_column response \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir $OUTPUT_DIR \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --num_train_epochs 2 \
    --logging_steps 30 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 300 \
    --save_total_limit 3 \
    --learning_rate $LR \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 \
    --tf32 True \
    --optim adamw_torch \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    "${DEEPSPEED_ARGS[@]}" \
    --report_to tensorboard \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_dropout 0.05 2>&1 | tee ${OUTPUT_DIR}/train.log
