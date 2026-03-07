import argparse
import json
import os
from typing import Dict, Iterable, List

from data_preprocess import build_prompt


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_chosen_and_rejected(response_obj: Dict) -> Dict[str, str]:
    role = response_obj.get("role", "assistant")
    if role == "search":
        arguments = response_obj.get("arguments", {})
        chosen = "调用检索参数：" + json.dumps(arguments, ensure_ascii=False)
        rejected = "调用检索参数：{" + '"name":"未知酒店"' + "}"
    else:
        chosen = str(response_obj.get("content", "")).strip()
        rejected = "抱歉，我不太确定，可能需要您提供更多信息。"
    return {"chosen": chosen, "rejected": rejected}


def _iter_pref_rows(sft_rows: Iterable[Dict], include_search: bool = False):
    for row in sft_rows:
        context = row.get("context")
        response = row.get("response")
        if context is None or response is None:
            continue

        if isinstance(response, str):
            response = json.loads(response)

        if (not include_search) and response.get("role") == "search":
            continue

        prompt = build_prompt(context)
        pref = _build_chosen_and_rejected(response)
        yield {
            "prompt": prompt,
            "chosen": pref["chosen"],
            "rejected": pref["rejected"],
        }


def _dump_jsonl(path: str, rows: Iterable[Dict]):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build weak DPO dataset from SFT jsonl files")
    parser.add_argument("--sft_train", type=str, default="../data/train.jsonl")
    parser.add_argument("--sft_dev", type=str, default="../data/dev.jsonl")
    parser.add_argument("--dpo_train", type=str, default="../data/dpo_train.jsonl")
    parser.add_argument("--dpo_dev", type=str, default="../data/dpo_dev.jsonl")
    parser.add_argument("--include_search", action="store_true", help="Include search/tool-call responses for DPO pairs")
    args = parser.parse_args()

    train_rows = _load_jsonl(args.sft_train)
    dev_rows = _load_jsonl(args.sft_dev)

    train_pref_rows = list(_iter_pref_rows(train_rows, include_search=args.include_search))
    dev_pref_rows = list(_iter_pref_rows(dev_rows, include_search=args.include_search))

    _dump_jsonl(args.dpo_train, train_pref_rows)
    _dump_jsonl(args.dpo_dev, dev_pref_rows)

    print(f"[Saved] dpo train -> {args.dpo_train} ({len(train_pref_rows)} rows)")
    print(f"[Saved] dpo dev   -> {args.dpo_dev} ({len(dev_pref_rows)} rows)")
    print("[Note] This is weakly-constructed preference data for bootstrapping DPO.")


if __name__ == "__main__":
    main()
