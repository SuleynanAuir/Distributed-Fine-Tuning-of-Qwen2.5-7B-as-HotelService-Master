from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to enable QLoRA (4-bit base model loading)."}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load base model with 4-bit quantization (same as use_qlora)."}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "bitsandbytes 4-bit quant type, e.g. nf4/fp4."}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use nested quantization for 4-bit."}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit model. one of: float16, bfloat16, float32"}
    )
    

@dataclass
class PeftArguments:
    # lora args
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank number"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha weight"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated target module names for LoRA."}
    )
    lora_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to LoRA checkpoints"}
    )
    # fourier args
    n_frequency: int = field(
        default=1000,
        metadata={"help": "the num_frequency of the Fourier adapters"}
    )
    scale: float = field(
        default=300.0,
        metadata={"help": "the scale of the Fourier adapters"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the history of chat."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
