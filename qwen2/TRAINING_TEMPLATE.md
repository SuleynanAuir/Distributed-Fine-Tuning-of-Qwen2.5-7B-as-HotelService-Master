# Qwen2.5 多 GPU 训练模板（2×RTX 4080 SUPER）

本目录已提供可直接运行的脚本：

- `train.sh`：当前默认 SFT LoRA（2 GPU + DeepSpeed ZeRO2）
- `train_sft_lora_2gpu.sh`：SFT LoRA 模板
- `train_sft_qlora_2gpu.sh`：SFT QLoRA 模板（4bit）
- `train_dpo_2gpu.sh`：DPO 模板（LoRA/QLoRA + ZeRO2）
- `deepspeed_zero2.json`：DeepSpeed ZeRO2 配置

## 1) 安装依赖

```bash
cd /root/autodl-tmp/fineTuningLab
pip install -r requirements.txt
```

## 2) SFT LoRA（推荐起点）

```bash
cd /root/autodl-tmp/fineTuningLab/qwen2
bash train_sft_lora_2gpu.sh
```

## 3) SFT QLoRA（显存更省）

```bash
cd /root/autodl-tmp/fineTuningLab/qwen2
bash train_sft_qlora_2gpu.sh
```

## 4) DPO（偏好优化）

DPO 需要数据是 JSONL，每行至少包含：

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

然后运行：

```bash
cd /root/autodl-tmp/fineTuningLab/qwen2
bash train_dpo_2gpu.sh
```

## 5) 4080S 显存不足时调参建议

- 优先把 `--per_device_train_batch_size` 从 `2` 降到 `1`
- 再把 `--max_source_length` 从 `2048` 降到 `1536` 或 `1024`
- 增大 `--gradient_accumulation_steps` 保持等效 batch
- 继续保留 `--gradient_checkpointing True`
- QLoRA 下优先使用 `--optim paged_adamw_32bit`

## 6) 日志与产物

- 输出目录：`qwen2/output/<run-name>-<timestamp>/`
- 训练日志：`train.log`
- LoRA checkpoint：`checkpoint-*`


----

# 📊 Model Evaluation Results

## Model Information

| Item | Value |
|-----|-----|
| Base Model | Qwen2.5-7B-Instruct |
| Fine-tuned Checkpoint | checkpoint-2150 |
| Dataset | `test.jsonl` |
| Total Samples | **560 Long Dialogue** |
| Search Samples | **242 search 🔍** |
| Assistant Samples | **318 assistant 🤖** |

---

# 🚀 Overall Performance

| Metric | Score |
|------|------|
| Role Accuracy | **🙆 99.11%** |
| Slot Precision | **95.96%** |
| Slot Recall | **95.50%** |
| Slot F1 | **✅ 95.73%** |
| Slot Exact Match | **90.08%** |
| BLEU-4 | **44.51** |
| ROUGE-L F1 | **60.40** |

---

# 📈 Metric Definitions

| Metric | Description |
|------|------|
| Role Accuracy | Accuracy of predicted role (`search` / `assistant`) |
| Slot Precision | Precision of predicted slots (search tasks) |
| Slot Recall | Recall of predicted slots (search tasks) |
| Slot F1 | F1 score of predicted slot values |
| Slot Exact Match | Exact match rate of predicted argument dictionaries |
| BLEU-4 | Character-level BLEU score for assistant responses |
| ROUGE-L F1 | Character-level ROUGE-L F1 score for assistant responses |

---

# 💬 Example Case

### Multi-turn Dialogue Scenario

User interacts with the system to search for hotels and finally confirms a booking.

---

### Conversation Context

```text
User: 恩，那好吧，再帮我找一个评分4分以上的酒店，在什么地方都行。

System → search:
{
  "rating_range_lower": 4.0
}

Return Result:
北京贵都大酒店 (评分 4.7)

Assistant:
北京贵都大酒店是个不错的选择。

User: 好的，最后帮我找一个酒店，要4.5分以上的，酒店有吹风机就好。

System → search:
{
  "facilities": ["吹风机"],
  "rating_range_lower": 4.5
}

Return Result:
北京富力万丽酒店 (评分 4.7)

Assistant:
北京富力万丽酒店呗！

User:
好的，我决定入住北京富力万丽酒店了！