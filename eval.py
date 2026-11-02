import re
import sys
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple, Callable
from tqdm import tqdm
import argparse
from traceback import print_exc

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import SYSTEM_PROMPT, _DIAG_SYS
from prompts_en import SYSTEM_PROMPT_en, _DIAG_SYS_en_WITH_A, _DIAG_SYS_en_WITHOUT_A
from postprocess import postprocess, canonicalize


def save_testdata():
    arrow_path = "/data1/med_group/grpo_trl/grpo_medical_v2/test_dataset/data-00000-of-00001.arrow"
    jsonl_path = "./data/test_data.jsonl"

    # 读取 arrow 文件
    dataset = Dataset.from_file(arrow_path)

    # 保存为 jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in dataset:
            prompt = item["prompt"]
            input = prompt.split('<|im_end|>\n<|im_start|>user')[-1].strip().split('<|im_end|>')[0]
            res = {
                "input": input,
                "true_diagnoses": eval(item["true_diagnoses"]),
            }
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Saved to {jsonl_path}")




'''
python eval.py \
  --test-jsonl c \
  --a-base-url http://127.0.0.1:30000/v1 \
  --b-base-url http://127.0.0.1:30001/v1 \
  --a-model-name default \
  --b-model-name default \
  --a-output-jsonl ./output/eval_reults/a_model_outputs.jsonl \
  --b-eval-output-jsonl ./output/eval_reults/b_model_eval_outputs.jsonl \
  --b-eval-metrics-json ./output/eval_reults/b_model_eval_metrics.json \
  --max-new-tokens-a 2048 \
  --max-new-tokens-b 512 \
  --temperature 0 \
  --a-concurrency 4 \
  --b-concurrency 4 \
  --max-retries 3 \
  --retry-sleep-seconds 2

# 已有A的结果，只重跑B
python eval_ab_sglang_async.py \
  --test-jsonl ./data/test_data.jsonl \
  --a-base-url http://127.0.0.1:30000/v1 \
  --b-base-url http://127.0.0.1:30001/v1 \
  --a-output-jsonl ./output/eval_reults/a_model_outputs.jsonl \
  --reuse-a-output


# test_data
CUDA_VISIBLE_DEVICES=1 python eval.py --test-jsonl ./data/test_data.jsonl \
    --a-model-path /data1/med_group/grpo_trl/grpo_medical_v2/checkpoint-8300 \
    --b-model-path /data_net/models_for_all/GLM-4-9B-0414 \
    --a-output-jsonl ./output/eval_results/a_model_outputs.jsonl \
    --b-eval-output-jsonl ./output/eval_results/GLM-4-9B-0414_eval_outputs.jsonl \
    --b-eval-metrics-json ./output/eval_results/GLM-4-9B-0414_eval_metrics.json \
    --reuse-a-output

# mimiciii
CUDA_VISIBLE_DEVICES=0 python eval.py --test-jsonl ./data/mimic_50.jsonl \
    --a-model-path /data1/med_group/grpo_trl/grpo_medical_v2/checkpoint-8300 \
    --a-output-jsonl ./output/eval_results_mimic/a_model_outputs.jsonl \
    --b-model-path /data_net/models_for_all/llama3.1-8B-Instruct \
    --b-eval-output-jsonl ./output/eval_results_mimic/llama3.1-8B-Instruct_eval_outputs_3.jsonl \
    --b-eval-metrics-json ./output/eval_results_mimic/llama3.1-8B-Instruct_eval_metrics_3.json \
    --reuse-a-output

# 只计算指标
python eval.py --metrics-only \
  --test-jsonl ./data/mimic_50.jsonl \
  --a-output-jsonl ./output/eval_results_mimic/a_model_outputs.jsonl \
  --b-eval-output-jsonl ./output/eval_results_mimic/llama3.1-8B-Instruct_eval_outputs_3.jsonl \
  --b-eval-metrics-json ./output/eval_results_mimic/llama3.1-8B-Instruct_eval_metrics_3_1.json
'''



# ============================================================
# 1. JSONL / JSON 工具
# ============================================================

def make_disease_only_filter(test_jsonl_path):
    """如果是 mimic 数据集，返回一个过滤 gold 的回调；否则返回 None。"""
    if 'mimic' not in str(test_jsonl_path):
        return None

    def _filter(row):
        gold = row.get("true_diagnoses", [])
        gold = [g for g in gold if canonicalize(g) is not None]
        if not gold:
            return None
        row["true_diagnoses"] = gold
        return row
    return _filter


def load_jsonl(
        path: Path,
        row_filter: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
    """加载 JSONL。可选 row_filter：
    - 返回 None 表示丢弃该样本
    - 返回 dict 表示保留（可就地修改，如缩窄 true_diagnoses）
    """
    data = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败：{path}, line={line_no}, error={e}") from e

            if row_filter is not None:
                row = row_filter(row)
                if row is None:
                    continue
            data.append(row)
    if row_filter is not None:
        print(f"load_jsonl: kept {len(data)} rows from {path}")       
    return data


def save_jsonl(data: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 2. Prompt 构造
# ============================================================

def parse_a_output(args, a_output: str) -> Dict[str, str]:
    """
    按 <DIAGNOSIS> 分块解析 A 模型输出。

    输出为：
    {
        "ee_block": "...",
        "ec_block": "..."
    }
    """
    if not a_output:
        return {
            "ee_block": "",
            "ec_block": "",
        }

    diagnosis_blocks = re.findall(
        r"<DIAGNOSIS>(.*?)</DIAGNOSIS>",
        a_output,
        flags=re.S,
    )

    # 如果没有完整 DIAGNOSIS 块，则退化为全局解析
    if not diagnosis_blocks:
        ee_items = re.findall(r"<EE>(.*?)</EE>", a_output, flags=re.S)
        ec_items = re.findall(r"<EC>(.*?)</EC>", a_output, flags=re.S)

        ee_block = "\n".join(
            item.strip()
            for item in ee_items
            if item.strip()
        )

        ec_block = "\n\n".join(
            item.strip()
            for item in ec_items
            if item.strip()
        )

        return {
            "ee_block": ee_block,
            "ec_block": ec_block,
        }

    ee_sections = []
    ec_sections = []

    for idx, block in enumerate(diagnosis_blocks, start=1):
        ee_items = re.findall(r"<EE>(.*?)</EE>", block, flags=re.S)
        ec_items = re.findall(r"<EC>(.*?)</EC>", block, flags=re.S)

        ee_items = [
            item.strip()
            for item in ee_items
            if item.strip()
        ]

        ec_items = [
            item.strip()
            for item in ec_items
            if item.strip()
        ]

        if ee_items:
            ee_sections.append(
                f"[Diagnostic candidate {idx}]\n" + "\n".join(ee_items) if 'mimic' in str(args.test_jsonl) else f"【诊断候选{idx}】\n" + "\n".join(ee_items)
            )

        if ec_items:
            ec_sections.append(
                f"[Diagnostic candidate {idx}]\n" + "\n\n".join(ec_items) if 'mimic' in str(args.test_jsonl) else f"【诊断候选{idx}】\n" + "\n\n".join(ec_items)
            )

    return {
        "ee_block": "\n\n".join(ee_sections),
        "ec_block": "\n\n".join(ec_sections),
    }


def build_a_prompt(args, case_input: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_en if 'mimic' in str(args.test_jsonl) else SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Medical Record:\n" + case_input if 'mimic' in str(args.test_jsonl) else "病历：\n" + case_input,
        },
    ]


def build_b_prompt_without_a(args, case_input: str) -> List[Dict[str, str]]:
    user_content = (
        f"[Medical Record Information]\n{case_input}\n\n"
        f"[Diagnosis Result]\n"
    ) if 'mimic' in str(args.test_jsonl) else (
        f"【病历信息】\n{case_input}\n\n"
        f"【诊断结果】\n"
    )

    return [
        {
            "role": "system",
            "content": _DIAG_SYS_en_WITHOUT_A if 'mimic' in str(args.test_jsonl) else _DIAG_SYS,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_b_prompt_with_a(args, case_input: str, a_output: str) -> List[Dict[str, str]]:
    parsed = parse_a_output(args, a_output)

    ee_block = parsed["ee_block"]
    ec_block = parsed["ec_block"]

    user_content = (
        f"[Medical Record Information]\n{case_input}\n\n"
        f"[Evidence Elements]\n{ee_block}\n\n"
        f"[Evidence Chain]\n{ec_block}\n\n"
        f"[Diagnosis Result]\n"
    ) if 'mimic' in str(args.test_jsonl) else (
        f"【病历信息】\n{case_input}\n\n"
        f"【证据元素】\n{ee_block}\n\n"
        f"【证据链】\n{ec_block}\n\n"
        f"【诊断结果】\n"
    ) 

    return [
        {
            "role": "system",
            "content": _DIAG_SYS_en_WITH_A if 'mimic' in str(args.test_jsonl) else _DIAG_SYS,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


# ============================================================
# 3. Transformers 本地模型加载与推理
# ============================================================

def load_local_model(
    model_path: str,
    torch_dtype: str = "float16",
):
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype='float16',
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model.eval()
    model = torch.compile(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def messages_to_input_ids(
    messages: List[Dict[str, str]],
    tokenizer,
    device,
):
    """
    优先使用 tokenizer.apply_chat_template。
    如果模型 tokenizer 没有 chat_template，则退化成简单拼接。
    """
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )
        input_ids = input_ids["input_ids"]
    except Exception as e:
        prompt_text = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt_text += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt_text += f"<|assistant|>\n{content}\n"
            else:
                prompt_text += f"{content}\n"

        prompt_text += "<|assistant|>\n"

        input_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
        )["input_ids"]

    return input_ids.to(device)


@torch.inference_mode()
def generate_text(
    messages: List[Dict[str, str]],
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> str:
    input_ids = messages_to_input_ids(
        messages=messages,
        tokenizer=tokenizer,
        device=model.device,
    )

    attention_mask = torch.ones_like(input_ids, device=model.device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0][input_ids.shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# 4. A 模型本地推理
# ============================================================

def run_a_model_local_one(
    sample: Dict[str, Any],
    idx: int,
    model,
    tokenizer,
    args,
) -> Dict[str, Any]:
    case_input = sample["input"]
    true_diagnoses = sample.get("true_diagnoses", [])

    messages = build_a_prompt(args, case_input)

    a_output = generate_text(
        messages=messages,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens_a,
        temperature=args.temperature,
    )

    return {
        "id": sample.get("id", idx),
        "input": case_input,
        "true_diagnoses": true_diagnoses,
        "a_output": a_output,
    }


def run_a_model_local(
    test_data: List[Dict[str, Any]],
    model,
    tokenizer,
    args,
) -> List[Dict[str, Any]]:
    results = []

    for idx, sample in enumerate(tqdm(test_data, desc="Running A model")):
        result = run_a_model_local_one(
            sample=sample,
            idx=idx,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        results.append(result)

    results.sort(key=lambda x: x["id"])

    save_jsonl(results, args.a_output_jsonl)
    print(f"A model outputs saved to: {args.a_output_jsonl}")

    return results


# ============================================================
# 5. B 输出解析
# ============================================================

# def extract_json_object(text: str) -> Dict[str, Any]:
#     text = text.strip()

#     text = re.sub(r"^```json\s*", "", text)
#     text = re.sub(r"^```\s*", "", text)
#     text = re.sub(r"\s*```$", "", text)

#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     match = re.search(r"\{.*\}", text, flags=re.S)
#     if match:
#         candidate = match.group(0)

#         try:
#             return json.loads(candidate)
#         except Exception:
#             pass

#     return {}


def parse_b_diagnoses(text: str) -> List[str]:
    # obj = extract_json_object(text)

    # diagnoses = obj.get("diagnoses", [])
    diagnoses = text.split('\n')

    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]

    if not isinstance(diagnoses, list):
        return []

    return [
        str(x).strip()
        for x in diagnoses
        if str(x).strip()
    ]


# ============================================================
# 6. 指标计算
# ============================================================

def normalize_diagnosis(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", "", text)

    text = text.replace("，", ",")
    text = text.replace("。", "")
    text = text.replace("；", ";")
    text = text.replace("：", ":")

    text = text.replace("Ca", "CA")
    text = text.replace("ca", "CA")
    text = text.replace("癌症", "癌")

    return text


def normalize_diagnosis_list(items: List[str]) -> Set[str]:
    return {
        normalize_diagnosis(x)
        for x in items
        if normalize_diagnosis(x)
    }


def compute_sample_metrics(
    pred: List[str],
    gold: List[str],
) -> Dict[str, Any]:
    """
    单条样本指标。
    新增字段：
      - hallucination_rate: 幻觉率 = FP / |pred|  （= 1 - precision）
      - fp_count:           当前样本预测错的疾病数（用于人均 FP）
      - fp_diseases:        当前样本被误报的疾病列表（用于条件幻觉率）
      - pred_diseases:      当前样本预测出的疾病列表（用于条件幻觉率分母）
    """
    pred_set = normalize_diagnosis_list(pred)
    gold_set = normalize_diagnosis_list(gold)

    # 两边都为空：视为完美预测，无 FP
    if not gold_set and not pred_set:
        return {
            "exact_match": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "hallucination_rate": 0.0,
            "fp_count": 0,
            "fp_diseases": [],
            "pred_diseases": [],
        }

    # 没有任何预测：precision/recall/f1 都是 0；幻觉率定义为 0（分母为 0）
    if not pred_set:
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hallucination_rate": None,  # 原来为0.0
            "fp_count": 0,
            "fp_diseases": [],
            "pred_diseases": [],
        }

    tp_set = pred_set & gold_set
    fp_set = pred_set - gold_set

    tp = len(tp_set)
    fp = len(fp_set)

    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    exact_match = 1.0 if pred_set == gold_set else 0.0

    # 指标4：幻觉率 = 预测错的 / 所有预测出的 = 1 - precision
    hallucination_rate = fp / len(pred_set)

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": hallucination_rate,
        "fp_count": fp,
        "fp_diseases": sorted(fp_set),
        "pred_diseases": sorted(pred_set),
    }


def compute_conditional_hallucination_rate(
    rows: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, Dict[str, float]]:
    """
    指标5：条件幻觉率 CHR(d)
        CHR(d) = 预测为 d 但实际没有 d 的病人数 / 预测为 d 的病人总数

    返回：
        {
            disease_name: {
                "chr": float,
                "pred_count": int,   # 预测为该疾病的病人数
                "fp_count":   int,   # 其中误报的病人数
            },
            ...
        }
    """
    pred_counter: Dict[str, int] = defaultdict(int)
    fp_counter: Dict[str, int] = defaultdict(int)

    for row in rows:
        metric_key = f"{mode}_metrics"
        if metric_key not in row:
            continue

        sample_metric = row[metric_key]
        pred_diseases = sample_metric.get("pred_diseases", [])
        fp_diseases = set(sample_metric.get("fp_diseases", []))

        for d in pred_diseases:
            pred_counter[d] += 1
            if d in fp_diseases:
                fp_counter[d] += 1

    chr_table: Dict[str, Dict[str, float]] = {}
    for disease, pred_count in pred_counter.items():
        fp_count = fp_counter.get(disease, 0)
        chr_table[disease] = {
            "chr": fp_count / pred_count if pred_count else 0.0,
            "pred_count": pred_count,
            "fp_count": fp_count,
        }

    # 按 chr 从高到低排序，方便快速定位易误报的疾病
    chr_table = dict(
        sorted(
            chr_table.items(),
            key=lambda kv: (-kv[1]["chr"], -kv[1]["pred_count"]),
        )
    )

    return chr_table


def aggregate_metrics(
    rows: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    """
    聚合指标：
      - 指标1 precision、指标2 recall、指标3 exact_match、f1、指标4 hallucination_rate
        均为样本级平均
      - 指标5 conditional_hallucination_rate：按疾病维度聚合
      - 指标6 mean_fp_per_patient = 总 FP 数 / 病人总数
    """
    mean_keys = [
        "exact_match",
        "precision",
        "recall",
        "f1",
        "hallucination_rate",
    ]

    result: Dict[str, Any] = {}

    metric_rows = [
        row[f"{mode}_metrics"]
        for row in rows
        if f"{mode}_metrics" in row
    ]

    for key in mean_keys:
        # values = [m[key] for m in metric_rows if key in m]
        values = [m[key] for m in metric_rows if m.get(key) is not None]
        result[f"{mode}_{key}"] = (
            sum(values) / len(values) if values else 0.0
        )

    # 指标6：平均每个病人的幻觉疾病数
    total_fp = sum(m.get("fp_count", 0) for m in metric_rows)
    n_patients = len(metric_rows)
    result[f"{mode}_mean_fp_per_patient"] = (
        total_fp / n_patients if n_patients else 0.0
    )
    result[f"{mode}_total_fp"] = total_fp
    result[f"{mode}_n_patients"] = n_patients

    # 指标5：条件幻觉率
    result[f"{mode}_conditional_hallucination_rate"] = (
        compute_conditional_hallucination_rate(rows, mode)
    )

    return result

# ============================================================
# 7. B 模型本地推理与评测
# ============================================================

def run_b_model_local_one(
    row: Dict[str, Any],
    model,
    tokenizer,
    args,
) -> Dict[str, Any]:
    sample_id = row["id"]
    case_input = row["input"]
    true_diagnoses = row.get("true_diagnoses", [])
    a_output = row["a_output"]

    messages_without_a = build_b_prompt_without_a(args, case_input)
    messages_with_a = build_b_prompt_with_a(args, case_input, a_output)

    b_output_without_a = generate_text(
        messages=messages_without_a,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens_b,
        temperature=args.temperature,
    )

    b_output_with_a = generate_text(
        messages=messages_with_a,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens_b,
        temperature=args.temperature,
    )

    is_mimic = 'mimic' in str(args.test_jsonl)
    pred_without_a = parse_b_diagnoses(b_output_without_a)
    pred_with_a = parse_b_diagnoses(b_output_with_a)
    pred_without_a = postprocess(pred_without_a, use_whitelist=is_mimic)
    pred_with_a    = postprocess(pred_with_a,    use_whitelist=is_mimic)

    # ↓↓↓ 新增这两行：gold 也过滤到 disease-only ↓↓↓
    # if is_mimic:
    #     true_diagnoses = [g for g in true_diagnoses if canonicalize(g) is not None]
    #     if not true_diagnoses:          # ← 新增：gold 为空则丢弃该样本
    #         return None

    metrics_without_a = compute_sample_metrics(
        pred=pred_without_a,
        gold=true_diagnoses,
    )

    metrics_with_a = compute_sample_metrics(
        pred=pred_with_a,
        gold=true_diagnoses,
    )

    return {
        "id": sample_id,
        "true_diagnoses": true_diagnoses,

        "a_output": a_output,

        "b_output_without_a": b_output_without_a,
        "pred_without_a": pred_without_a,
        "without_a_metrics": metrics_without_a,

        "b_output_with_a": b_output_with_a,
        "pred_with_a": pred_with_a,
        "with_a_metrics": metrics_with_a,
    }


def run_b_model_and_eval_local(
    a_results: List[Dict[str, Any]],
    model,
    tokenizer,
    args,
) -> Dict[str, Any]:
    eval_rows = []

    for row in tqdm(a_results, desc="Running B model"):
        result = run_b_model_local_one(
            row=row,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        if result is None:              # ← 新增
            continue
        eval_rows.append(result)

    eval_rows.sort(key=lambda x: x["id"])

    save_jsonl(eval_rows, args.b_eval_output_jsonl)
    print(f"B model eval outputs saved to: {args.b_eval_output_jsonl}")

    metrics = {}

    metrics.update(aggregate_metrics(eval_rows, "without_a"))
    metrics.update(aggregate_metrics(eval_rows, "with_a"))

    metrics["delta_exact_match"] = (
        metrics["with_a_exact_match"] - metrics["without_a_exact_match"]
    )
    metrics["delta_precision"] = (
        metrics["with_a_precision"] - metrics["without_a_precision"]
    )
    metrics["delta_recall"] = (
        metrics["with_a_recall"] - metrics["without_a_recall"]
    )
    metrics["delta_f1"] = (
        metrics["with_a_f1"] - metrics["without_a_f1"]
    )
    metrics["delta_hallucination_rate"] = (
        metrics["with_a_hallucination_rate"]
        - metrics["without_a_hallucination_rate"]
    )
    metrics["delta_mean_fp_per_patient"] = (
        metrics["with_a_mean_fp_per_patient"]
        - metrics["without_a_mean_fp_per_patient"]
    )

    save_json(metrics, args.b_eval_metrics_json)

    print(f"Metrics saved to: {args.b_eval_metrics_json}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    return metrics


# ============================================================
# 8. 仅计算指标（不跑模型推理）
# ============================================================

def compute_metrics_only(
    test_jsonl: Path,
    a_output_jsonl: Path,
    b_eval_output_jsonl: Path,
    b_eval_metrics_json: Path,
) -> Dict[str, Any]:
    """
    只根据已有文件重新计算指标，不做任何模型推理。
    输入：
      - test_jsonl:            测试集（提供 true_diagnoses 兜底）
      - a_output_jsonl:        A 模型输出（可选，用于回填 a_output）
      - b_eval_output_jsonl:   B 模型评测输出（包含 b_output_without_a / b_output_with_a）
    输出：
      - b_eval_metrics_json:   聚合指标 JSON
    """
    test_rows = load_jsonl(test_jsonl, row_filter=make_disease_only_filter(test_jsonl))
    test_by_id = {row.get("id", i): row for i, row in enumerate(test_rows)}

    a_by_id: Dict[Any, Dict[str, Any]] = {}
    if a_output_jsonl and Path(a_output_jsonl).exists():
        for row in load_jsonl(a_output_jsonl, row_filter=make_disease_only_filter(test_jsonl)):
            a_by_id[row.get("id")] = row

    b_rows = load_jsonl(b_eval_output_jsonl, row_filter=make_disease_only_filter(test_jsonl))

    eval_rows = []
    for row in b_rows:
        sample_id = row.get("id")
        true_diagnoses = (
            row.get("true_diagnoses")
            or a_by_id.get(sample_id, {}).get("true_diagnoses")
            or test_by_id.get(sample_id, {}).get("true_diagnoses")
            or []
        )

        b_output_without_a = row.get("b_output_without_a", "")
        b_output_with_a = row.get("b_output_with_a", "")

        # pred_without_a = row.get("pred_without_a") or parse_b_diagnoses(b_output_without_a)
        # pred_with_a = row.get("pred_with_a") or parse_b_diagnoses(b_output_with_a)
        is_mimic = 'mimic' in str(test_jsonl)
        pred_without_a = postprocess(parse_b_diagnoses(b_output_without_a), use_whitelist=is_mimic)
        pred_with_a    = postprocess(parse_b_diagnoses(b_output_with_a),    use_whitelist=is_mimic)

        # ↓↓↓ 新增这两行 ↓↓↓
        # if is_mimic:
        #     true_diagnoses = [g for g in true_diagnoses if canonicalize(g) is not None]
        #     if not true_diagnoses:          # ← 新增：gold 为空则跳过
        #         continue

        new_row = dict(row)
        new_row["true_diagnoses"] = true_diagnoses
        new_row["pred_without_a"] = pred_without_a
        new_row["pred_with_a"] = pred_with_a
        new_row["without_a_metrics"] = compute_sample_metrics(pred_without_a, true_diagnoses)
        new_row["with_a_metrics"] = compute_sample_metrics(pred_with_a, true_diagnoses)
        eval_rows.append(new_row)

    eval_rows.sort(key=lambda x: x.get("id", 0))

    metrics: Dict[str, Any] = {}
    metrics.update(aggregate_metrics(eval_rows, "without_a"))
    metrics.update(aggregate_metrics(eval_rows, "with_a"))
    for k in ["exact_match", "precision", "recall", "f1",
              "hallucination_rate", "mean_fp_per_patient"]:
        metrics[f"delta_{k}"] = metrics[f"with_a_{k}"] - metrics[f"without_a_{k}"]

    save_json(metrics, Path(b_eval_metrics_json))
    print(f"Metrics saved to: {b_eval_metrics_json}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


# ============================================================
# 9. 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate diagnosis models A/B locally via transformers."
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="不做模型推理，仅根据已有文件重算指标并写 b-eval-metrics-json。",
    )

    # 输入输出
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        required=True,
        help="测试集 JSONL 文件路径，每行包含 input 和 true_diagnoses 字段。",
    )
    parser.add_argument(
        "--a-output-jsonl",
        type=Path,
        default=Path("./output/eval_results/a_model_outputs.jsonl"),
        help="A 模型输出保存路径；reuse-a-output 时从这里读取。",
    )
    parser.add_argument(
        "--b-eval-output-jsonl",
        type=Path,
        default=Path("./output/eval_results/b_model_eval_outputs.jsonl"),
        help="B 模型两种模式的输出保存路径。",
    )
    parser.add_argument(
        "--b-eval-metrics-json",
        type=Path,
        default=Path("./output/eval_results/b_model_eval_metrics.json"),
        help="B 模型评测指标保存路径。",
    )

    # 本地模型路径
    parser.add_argument(
        "--a-model-path",
        type=str,
        default=None,
        help="A 模型本地路径。若指定 --reuse-a-output，则可以不传。",
    )
    parser.add_argument(
        "--b-model-path",
        type=str,
        required=True,
        help="B 模型本地路径。",
    )

    # 生成参数
    parser.add_argument(
        "--max-new-tokens-a",
        type=int,
        default=2048,
        help="A 模型最大生成 token 数。",
    )
    parser.add_argument(
        "--max-new-tokens-b",
        type=int,
        default=512,
        help="B 模型最大生成 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度。0 表示贪心解码。",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="模型加载精度。",
    )

    # 调试与复用
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只测试前 N 条样本，调试用。",
    )
    parser.add_argument(
        "--reuse-a-output",
        action="store_true",
        help="如果 A 输出文件已存在，则直接复用，不重新跑 A，只重跑 B。",
    )

    args = parser.parse_args()

    if not args.reuse_a_output and not args.a_model_path:
        raise ValueError(
            "未指定 --reuse-a-output 时，必须传入 --a-model-path。"
        )

    if args.reuse_a_output and not args.a_output_jsonl.exists():
        raise FileNotFoundError(
            f"指定了 --reuse-a-output，但 A 输出文件不存在：{args.a_output_jsonl}"
        )

    return args



# ============================================================
# 10. 主入口
# ============================================================

def main():
    # 快速路径：仅计算指标
    if "--metrics-only" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--metrics-only", action="store_true")
        p.add_argument("--test-jsonl", type=Path, required=True)
        p.add_argument("--a-output-jsonl", type=Path, required=True)
        p.add_argument("--b-eval-output-jsonl", type=Path, required=True)
        p.add_argument("--b-eval-metrics-json", type=Path, required=True)
        a = p.parse_args()
        compute_metrics_only(a.test_jsonl, a.a_output_jsonl,
                             a.b_eval_output_jsonl, a.b_eval_metrics_json)
        return

    # 运行模型推理、计算指标
    args = parse_args()

    start_time = time.time()

    test_data = load_jsonl(args.test_jsonl, row_filter=make_disease_only_filter(args.test_jsonl))

    if args.limit is not None:
        test_data = test_data[:args.limit]

    print(f"Loaded test samples: {len(test_data)}")

    if not test_data:
        raise ValueError("测试集为空。")

    # ========================================================
    # A 阶段：可复用已有 A 输出
    # ========================================================
    if args.reuse_a_output:
        print(f"Reusing existing A outputs: {args.a_output_jsonl}")
        a_results = load_jsonl(args.a_output_jsonl, row_filter=make_disease_only_filter(args.test_jsonl))

        if args.limit is not None:
            a_results = a_results[:args.limit]

        print(f"Loaded A outputs: {len(a_results)}")

    else:
        print("Loading A model...")

        a_model, a_tokenizer = load_local_model(
            model_path=args.a_model_path,
            torch_dtype=args.torch_dtype,
        )

        a_results = run_a_model_local(
            test_data=test_data,
            model=a_model,
            tokenizer=a_tokenizer,
            args=args,
        )

        # 释放 A 模型显存，给 B 模型腾空间
        del a_model
        del a_tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # B 阶段：每次都会重跑
    # ========================================================
    print("Loading B model...")

    b_model, b_tokenizer = load_local_model(
        model_path=args.b_model_path,
        torch_dtype=args.torch_dtype,
    )

    run_b_model_and_eval_local(
        a_results=a_results,
        model=b_model,
        tokenizer=b_tokenizer,
        args=args,
    )

    elapsed = time.time() - start_time
    print(f"Total elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()