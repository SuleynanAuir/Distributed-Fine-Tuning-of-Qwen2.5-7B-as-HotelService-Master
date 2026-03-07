MODEL_DIR="/root/autodl-tmp/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR="../qwen2/output/hotel_qwen2-20241103-132703/checkpoint-2150"

CUDA_VISIBLE_DEVICES=0 python webui_qwen2.py \
    --model $MODEL_DIR \
    --ckpt $CHECKPOINT_DIR
