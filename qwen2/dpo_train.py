import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import torch

if hasattr(torch, "amp") and not hasattr(torch.amp, "custom_fwd"):
    def _compat_custom_fwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        return torch.cuda.amp.custom_fwd(*args, **kwargs)

    torch.amp.custom_fwd = _compat_custom_fwd
if hasattr(torch, "amp") and not hasattr(torch.amp, "custom_bwd"):
    def _compat_custom_bwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        return torch.cuda.amp.custom_bwd(*args, **kwargs)

    torch.amp.custom_bwd = _compat_custom_bwd

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from data_preprocess import build_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training template for Qwen2.5 with LoRA/QLoRA")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=f"output/dpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--deepspeed", type=str, default=None)

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=8e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--min_response_tokens", type=int, default=4)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    dtype_name = (dtype_name or "bfloat16").lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def load_preference_jsonl(path: str, tokenizer, min_response_tokens: int = 4) -> Dataset:
    records: List[Dict] = []
    dropped_empty = 0
    dropped_same = 0
    dropped_short = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if "prompt" in row and "chosen" in row and "rejected" in row:
                prompt = row["prompt"]
                chosen = row["chosen"]
                rejected = row["rejected"]
            elif "context" in row and "chosen" in row and "rejected" in row:
                prompt = build_prompt(row["context"])
                chosen = row["chosen"]
                rejected = row["rejected"]
            else:
                raise ValueError(
                    "DPO data requires fields: (prompt, chosen, rejected) or (context, chosen, rejected)."
                )

            prompt = str(prompt).strip()
            chosen = str(chosen).strip()
            rejected = str(rejected).strip()

            if (not prompt) or (not chosen) or (not rejected):
                dropped_empty += 1
                continue
            if chosen == rejected:
                dropped_same += 1
                continue

            chosen_len = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
            rejected_len = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
            if chosen_len < min_response_tokens or rejected_len < min_response_tokens:
                dropped_short += 1
                continue

            records.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    if not records:
        raise ValueError(f"No valid records found in {path}")

    print(
        f"[Data] {path}: kept={len(records)}, "
        f"dropped_empty={dropped_empty}, dropped_same={dropped_same}, dropped_short={dropped_short}"
    )
    return Dataset.from_list(records)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = args.logging_dir or os.path.join(args.output_dir, "runs")
    os.makedirs(logging_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = resolve_dtype(args.bnb_4bit_compute_dtype)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model_kwargs = {"trust_remote_code": True}
    model_kwargs["low_cpu_mem_usage"] = True
    if args.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
        )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.use_gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_dataset = load_preference_jsonl(args.train_file, tokenizer, min_response_tokens=args.min_response_tokens)
    eval_dataset = (
        load_preference_jsonl(args.validation_file, tokenizer, min_response_tokens=args.min_response_tokens)
        if args.validation_file else None
    )

    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and args.max_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    print(f"[Data] train size={len(train_dataset)}")
    if eval_dataset is not None:
        print(f"[Data] eval size={len(eval_dataset)}")

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_steps=args.max_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        remove_unused_columns=False,
        gradient_checkpointing=args.use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.use_gradient_checkpointing else None,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        logging_dir=logging_dir,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=None,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
