import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_preprocess import build_prompt, parse_json


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_for_compare(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 6)
    if isinstance(value, list):
        normalized_list = [normalize_for_compare(v) for v in value]
        try:
            return sorted(normalized_list, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
        except Exception:
            return normalized_list
    if isinstance(value, dict):
        return {k: normalize_for_compare(v) for k, v in sorted(value.items(), key=lambda item: item[0])}
    return value


def flatten_slot_values(slot_dict: Optional[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    if not slot_dict:
        return []
    flattened: List[Tuple[str, Any]] = []
    for key in sorted(slot_dict.keys()):
        value = slot_dict[key]
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                flattened.append((key, normalize_for_compare(item)))
        else:
            flattened.append((key, normalize_for_compare(value)))
    return flattened


def compute_slot_counts(
    pred_slots: Optional[Dict[str, Any]],
    gold_slots: Optional[Dict[str, Any]],
) -> Tuple[int, int, int, bool]:
    pred_items = flatten_slot_values(pred_slots)
    gold_items = flatten_slot_values(gold_slots)
    pred_set = set(pred_items)
    gold_set = set(gold_items)
    correct = len(pred_set & gold_set)
    pred_count = len(pred_set)
    gold_count = len(gold_set)

    pred_norm = normalize_for_compare(pred_slots or {})
    gold_norm = normalize_for_compare(gold_slots or {})
    exact_match = pred_norm == gold_norm
    return correct, pred_count, gold_count, exact_match


def bleu4_char_level(pred_text: str, ref_text: str) -> float:
    pred_text = (pred_text or "").strip()
    ref_text = (ref_text or "").strip()
    if not pred_text or not ref_text:
        return 0.0
    return sentence_bleu(
        [list(ref_text)],
        list(pred_text),
        smoothing_function=SmoothingFunction().method3,
    )


def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_f1_char_level(pred_text: str, ref_text: str) -> float:
    pred_chars = list((pred_text or "").strip())
    ref_chars = list((ref_text or "").strip())
    if not pred_chars or not ref_chars:
        return 0.0
    lcs = lcs_length(pred_chars, ref_chars)
    precision = lcs / len(pred_chars)
    recall = lcs / len(ref_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("<|im_start|>assistant"):
        text = text.replace("<|im_start|>assistant", "", 1).strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].strip()
    return text


def extract_search_arguments(text: str) -> Optional[Dict[str, Any]]:
    parsed = parse_json(text)
    if parsed is None:
        return None
    if isinstance(parsed, dict) and "arguments" in parsed and isinstance(parsed["arguments"], dict):
        return parsed["arguments"]
    if isinstance(parsed, dict):
        return parsed
    return None


def resolve_lora_ckpt_path(ckpt_path: Optional[str]) -> Optional[str]:
    if not ckpt_path:
        return None

    ckpt_path = os.path.abspath(ckpt_path)
    adapter_config = os.path.join(ckpt_path, "adapter_config.json")
    if os.path.isfile(adapter_config):
        return ckpt_path

    if not os.path.isdir(ckpt_path):
        raise ValueError(f"LoRA checkpoint path not found: {ckpt_path}")

    candidate_dirs: List[Tuple[int, str]] = []
    for child in os.listdir(ckpt_path):
        full_path = os.path.join(ckpt_path, child)
        if not os.path.isdir(full_path):
            continue
        if not child.startswith("checkpoint-"):
            continue
        try:
            step = int(child.split("-")[-1])
        except ValueError:
            continue
        if os.path.isfile(os.path.join(full_path, "adapter_config.json")):
            candidate_dirs.append((step, full_path))

    if not candidate_dirs:
        raise ValueError(
            f"Can't find 'adapter_config.json' under '{ckpt_path}'. "
            "Please pass a valid LoRA checkpoint directory (e.g. .../checkpoint-2150)."
        )

    candidate_dirs.sort(key=lambda x: x[0], reverse=True)
    best_ckpt = candidate_dirs[0][1]
    print(f"[Info] Auto-resolved LoRA checkpoint: {best_ckpt}")
    return best_ckpt


def load_model_and_tokenizer(model_path: str, ckpt_path: Optional[str], device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    resolved_ckpt_path = resolve_lora_ckpt_path(ckpt_path)
    if resolved_ckpt_path:
        model = PeftModel.from_pretrained(model, model_id=resolved_ckpt_path)
    model = model.to(device).eval()
    return tokenizer, model, resolved_ckpt_path


def run_generation(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = load_jsonl(args.data)
    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    tokenizer, model, resolved_ckpt_path = load_model_and_tokenizer(args.model, args.ckpt, args.device)

    total = len(dataset)
    search_total = 0
    assistant_total = 0
    role_correct = 0

    correct_slots = 0
    pred_slots = 0
    gold_slots = 0
    slot_exact_match_count = 0

    bleu_scores: List[float] = []
    rouge_l_scores: List[float] = []

    prediction_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating"), start=1):
        prompt = build_prompt(item["context"])
        pred_raw = run_generation(tokenizer, model, prompt, args.device, args.max_new_tokens)
        pred_clean = clean_model_text(pred_raw)

        label_obj = item["response"]
        if isinstance(label_obj, str):
            label_obj = json.loads(label_obj)
        gold_role = label_obj.get("role", "")

        row: Dict[str, Any] = {
            "index": idx,
            "expected_role": gold_role,
            "context": item["context"],
            "reference": label_obj,
            "prediction_text": pred_clean,
        }

        if gold_role == "search":
            search_total += 1
            pred_args = extract_search_arguments(pred_clean)
            pred_role = "search" if pred_args is not None else "assistant"
            if pred_role == gold_role:
                role_correct += 1

            gold_args = label_obj.get("arguments", {})
            c, p, g, exact = compute_slot_counts(pred_args or {}, gold_args)
            correct_slots += c
            pred_slots += p
            gold_slots += g
            slot_exact_match_count += int(exact)

            row.update(
                {
                    "predicted_role": pred_role,
                    "predicted_arguments": pred_args,
                    "gold_arguments": gold_args,
                    "slot_correct": c,
                    "slot_pred_total": p,
                    "slot_gold_total": g,
                    "slot_exact_match": exact,
                }
            )
        else:
            assistant_total += 1
            pred_role = "search" if extract_search_arguments(pred_clean) is not None else "assistant"
            if pred_role == gold_role:
                role_correct += 1

            gold_text = label_obj.get("content", "")
            bleu = bleu4_char_level(pred_clean, gold_text)
            rouge_l = rouge_l_f1_char_level(pred_clean, gold_text)
            bleu_scores.append(bleu)
            rouge_l_scores.append(rouge_l)

            row.update(
                {
                    "predicted_role": pred_role,
                    "gold_text": gold_text,
                    "bleu4": round(bleu * 100, 4),
                    "rouge_l_f1": round(rouge_l * 100, 4),
                }
            )

        prediction_rows.append(row)

    slot_p = (correct_slots / pred_slots) if pred_slots > 0 else 0.0
    slot_r = (correct_slots / gold_slots) if gold_slots > 0 else 0.0
    slot_f1 = (2 * slot_p * slot_r / (slot_p + slot_r)) if (slot_p + slot_r) > 0 else 0.0

    metrics = {
        "data_path": args.data,
        "model_path": args.model,
        "ckpt_path": resolved_ckpt_path,
        "total_samples": total,
        "search_samples": search_total,
        "assistant_samples": assistant_total,
        "role_accuracy": round((role_correct / total) * 100, 4) if total > 0 else 0.0,
        "slot_precision": round(slot_p * 100, 4),
        "slot_recall": round(slot_r * 100, 4),
        "slot_f1": round(slot_f1 * 100, 4),
        "slot_exact_match": round((slot_exact_match_count / search_total) * 100, 4) if search_total > 0 else 0.0,
        "bleu4": round((sum(bleu_scores) / len(bleu_scores)) * 100, 4) if bleu_scores else 0.0,
        "rouge_l_f1": round((sum(rouge_l_scores) / len(rouge_l_scores)) * 100, 4) if rouge_l_scores else 0.0,
        "scoring_standard": {
            "role_accuracy": "预测角色(search/assistant)与标注一致率，越高越好",
            "slot_f1": "仅对search样本，按槽位值集合计算F1，越高越好",
            "slot_exact_match": "仅对search样本，参数字典完全一致率，越高越好",
            "bleu4": "仅对assistant样本，字符级BLEU-4，越高越好",
            "rouge_l_f1": "仅对assistant样本，字符级ROUGE-L F1，越高越好",
        },
    }

    metrics_dir = os.path.dirname(args.metrics_out)
    predictions_dir = os.path.dirname(args.predictions_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    if predictions_dir:
        os.makedirs(predictions_dir, exist_ok=True)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(args.predictions_out, "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] metrics -> {args.metrics_out}")
    print(f"[Saved] predictions -> {args.predictions_out}")

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Evaluate model on test.jsonl with quantitative metrics and per-sample answers")
    parser.add_argument("--model", type=str, required=True, help="Base model path, e.g. /root/autodl-tmp/Qwen2.5-7B-Instruct")
    parser.add_argument("--ckpt", type=str, default=None, help="LoRA checkpoint path (optional)")
    parser.add_argument("--data", type=str, default="../data/test.jsonl", help="Test dataset jsonl path")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation max new tokens")
    parser.add_argument("--max_samples", type=int, default=0, help="Only evaluate first N samples, 0 means full set")
    parser.add_argument("--metrics_out", type=str, default=f"output/eval-metrics-{now}.json", help="Output metrics json path")
    parser.add_argument("--predictions_out", type=str, default=f"output/eval-predictions-{now}.jsonl", help="Output predictions jsonl path")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    evaluate(cli_args)
