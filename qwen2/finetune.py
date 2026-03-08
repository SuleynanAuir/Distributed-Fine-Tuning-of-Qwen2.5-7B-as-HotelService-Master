# 导入所需模块和库，包含用于加载模型、配置低秩适应（LoRA）参数、定义数据预处理等功能
import json
import os
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

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq, 
    HfArgumentParser,
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from data_preprocess import InputOutputDataset


def _resolve_dtype(dtype_name: str):
    dtype_name = (dtype_name or "bfloat16").lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def _parse_target_modules(target_modules: str):
    if not target_modules:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return [module.strip() for module in target_modules.split(",") if module.strip()]

def main():
    # 使用 HfArgumentParser 解析命令行参数，并将参数解析成数据类对象：model_args（模型相关）、data_args（数据相关）、peft_args（LoRA参数）、training_args（训练配置）
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    use_qlora = bool(model_args.use_qlora or model_args.load_in_4bit)
    compute_dtype = _resolve_dtype(model_args.bnb_4bit_compute_dtype)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    quantization_config = None
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    attn_impl = os.environ.get("ATTN_IMPLEMENTATION", "").strip()
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # 加载预训练的生成式语言模型 (AutoModelForCausalLM) 和分词器 (AutoTokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    if training_args.gradient_checkpointing and getattr(training_args, "gradient_checkpointing_kwargs", None) is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # 设置LoRA的配置
    lora_config = LoraConfig(
        inference_mode=False,
        # 指定任务类型为生成语言模型 (TaskType.CAUSAL_LM)
        task_type=TaskType.CAUSAL_LM,
        # 指定了模型中应用 LoRA 的模块，如 q_proj、k_proj 和 v_proj
        target_modules=_parse_target_modules(peft_args.lora_target_modules),
        r=peft_args.lora_rank, 
        lora_alpha=peft_args.lora_alpha, 
        lora_dropout=peft_args.lora_dropout
    )
    # 将 LoRA 配置应用到模型中，设备由 Trainer/Accelerate 在分布式模式下自动放置
    model = get_peft_model(model, lora_config)
    # 输出模型中的可训练参数数量
    model.print_trainable_parameters()

    # 设置数据规整器用于在训练过程中对数据批量填充
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True
    )

    # 如果启用了 do_train 标志，读取训练数据文件（JSONL 格式）并加载为列表格式，然后通过 InputOutputDataset 类预处理数据
    if training_args.do_train:
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
        train_dataset = InputOutputDataset(train_data, tokenizer, data_args)
    # 如果启用了 do_eval 标志，类似地读取验证数据文件并加载为验证数据集
    if training_args.do_eval:
        with open(data_args.validation_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]
        eval_dataset = InputOutputDataset(eval_data, tokenizer, data_args)

    # 实例化 Trainer 对象，用于训练和评估。传入模型、分词器、数据规整器、训练参数以及（如果启用训练或评估）数据集
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
    )

    # 启用梯度检查点来降低显存使用，并开启输入梯度需求，以便更高效的梯度计算
    if training_args.do_train:
        model.config.use_cache = False
        if not use_qlora:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        trainer.train()

    # 如果启用评估标志，调用 trainer.evaluate() 执行评估
    if training_args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()
