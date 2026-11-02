import argparse
import time
from pathlib import Path

from tqdm import tqdm

from eval_v2 import (
    is_mimic,
    make_gold_filter,
    load_jsonl,
    save_jsonl,
    save_json,
    load_local_model,
    generate_text,
    build_b_prompt_without_a,
    parse_b_diagnoses,
    compute_sample_metrics,
    aggregate_metrics,
    postprocess,
)


def run_e2e_one(sample, idx, model, tokenizer, args):
    case_input = sample["input"]
    true_diagnoses = sample["true_diagnoses"]

    # 直接使用病历，不经过 A 模型、EE、EC、证据链
    messages = build_b_prompt_without_a(
        args,
        case_input,
    )

    output = generate_text(
        messages,
        model,
        tokenizer,
        args.model_path,
        args.max_new_tokens,
        args.temperature,
        args.use_dola,
    )

    pred = postprocess(
        parse_b_diagnoses(output),
        use_whitelist=is_mimic(args.test_jsonl),
    )

    metrics = compute_sample_metrics(
        pred,
        true_diagnoses,
    )

    return {
        "id": sample.get("id", idx),
        "input": case_input,
        "true_diagnoses": true_diagnoses,
        "e2e_output": output,
        "pred_e2e": pred,
        "e2e_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-jsonl",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="端到端模型 checkpoint",
    )

    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "./output/e2e_eval/e2e_outputs.jsonl"
        ),
    )

    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path(
            "./output/e2e_eval/e2e_metrics.json"
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--use-dola",
        action="store_true",
    )

    args = parser.parse_args()

    start_time = time.time()

    # -----------------------------
    # 数据
    # -----------------------------
    gold_filter = make_gold_filter(
        args.test_jsonl
    )

    test_data = load_jsonl(
        args.test_jsonl,
        row_filter=gold_filter,
    )

    if args.limit is not None:
        test_data = test_data[:args.limit]

    print(
        f"Loaded test samples: {len(test_data)}"
    )

    # -----------------------------
    # 加载端到端模型
    # -----------------------------
    print(
        f"Loading E2E model: {args.model_path}"
    )

    model, tokenizer = load_local_model(
        args.model_path,
        args.torch_dtype,
    )

    # -----------------------------
    # 直接推理
    # -----------------------------
    rows = []

    for idx, sample in enumerate(
        tqdm(
            test_data,
            desc="Running E2E model",
        )
    ):
        row = run_e2e_one(
            sample,
            idx,
            model,
            tokenizer,
            args,
        )

        rows.append(row)

    # -----------------------------
    # 保存预测结果
    # -----------------------------
    rows.sort(
        key=lambda x: x["id"]
    )

    save_jsonl(
        rows,
        args.output_jsonl,
    )

    # -----------------------------
    # 计算整体指标
    # -----------------------------
    metrics = aggregate_metrics(
        rows,
        "e2e",
    )

    save_json(
        metrics,
        args.metrics_json,
    )

    print(
        f"E2E outputs saved to: "
        f"{args.output_jsonl}"
    )

    print(
        f"E2E metrics saved to: "
        f"{args.metrics_json}"
    )

    print(
        f"Total elapsed: "
        f"{time.time() - start_time:.2f}s"
    )



'''
CUDA_VISIBLE_DEVICES=3 python eval_sft.py \
    --test-jsonl ./data/mimic_50.jsonl \
    --model-path /data1/med_group/LLamaFactory/experiments/outputs/evidence_chain/sft_glm_with_ee \
    --output-jsonl ./output/eval_sft_results_mimic/sft_glm_with_ee_outputs.jsonl \
    --metrics-json ./output/eval_sft_results_mimic/sft_glm_with_ee_metrics.json
'''


if __name__ == "__main__":
    main()
