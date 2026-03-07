#! /usr/bin/env bash

set -ex

RUN_NAME=hotel_qwen25_dpo_2gpu
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"
DEEPSPEED_CONFIG="deepspeed_zero2.json"

# 请准备DPO格式数据，字段需包含: prompt/chosen/rejected
TRAIN_DPO_FILE="../data/dpo_train.jsonl"
DEV_DPO_FILE="../data/dpo_dev.jsonl"

if [[ ! -f "$TRAIN_DPO_FILE" || ! -f "$DEV_DPO_FILE" ]]; then
    echo "[Info] DPO data not found, auto-building from SFT data..."
    python build_dpo_from_sft.py \
        --sft_train ../data/train.jsonl \
        --sft_dev ../data/dev.jsonl \
        --dpo_train "$TRAIN_DPO_FILE" \
        --dpo_dev "$DEV_DPO_FILE"
fi

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=2 --master_port=29513 dpo_train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --train_file $TRAIN_DPO_FILE \
    --validation_file $DEV_DPO_FILE \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/runs \
    --report_to tensorboard \
    --deepspeed $DEEPSPEED_CONFIG \
    --use_qlora \
    --bnb_4bit_use_double_quant \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 8e-7 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --beta 0.08 \
    --max_grad_norm 0.5 \
    --use_gradient_checkpointing \
    --min_response_tokens 12 \
    --max_prompt_length 384 \
    --max_length 512 \
    --logging_steps 20 \
    --save_steps 200 \
    --eval_steps 100 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj 2>&1 | tee ${OUTPUT_DIR}/train.log
