"""
GRPO Medical Training Script  v2
目标：训练模型生成高质量的医疗"证据元素"(EE)和"证据链"(EC)

设计原则（v2 简化）：
  - 四路奖励拆成四个独立函数，TRL 自动记录每路的 mean/std，无需自定义 metrics callback
  - GRPOConfig.log_completions=True 原生打印生成样本，无需自定义 sample callback
  - eval_dataset 传给 Trainer 后，每隔 eval_steps 自动跑验证集并用相同 reward 函数评分
  - 模型以字符串路径传入，model_init_kwargs 放在 GRPOConfig 里并设 device_map=None，
    避免多卡 DDP 下 device_map='auto' 冲突

数据说明：
  processed_dataset.jsonl  —— 病历原文，字段：患者id / input / 标签列表
  annotated_results.jsonl  —— 标注结果，字段：患者id / 诊断列表（每个诊断含证据元素列表、证据链）
  两文件通过"患者id"合并，每条样本可能含多个诊断。

策略模型输出格式（不含诊断名称）：
  <DIAGNOSIS>
  <EE>直接摘自病历的证据元素</EE>
  ...
  <EC>证据推理链</EC>
  </DIAGNOSIS>
"""

import os
import re
import json
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import requests
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from openai import OpenAI

from prompts import SYSTEM_PROMPT, _DIAG_SYS, _COMP_SYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpo_medical_v2")


transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()

# ═══════════════════════════════════════════════════════════
#  1. 配置
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    model_name_or_path: str = "/data_net/models_for_all/Qwen2.5-0.5B"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

@dataclass
class CurriculumConfig:
    """课程式权重配置"""
    enable: bool = True
    init_weight_diagnosis:    float = 0.1
    init_weight_completeness: float = 0.1
    init_weight_faithfulness: float = 0.4
    init_weight_format:       float = 0.4

    target_weight_diagnosis:    float = 0.4
    target_weight_completeness: float = 0.25
    target_weight_faithfulness: float = 0.2
    target_weight_format:       float = 0.15
    
    warmup_steps: int = 200
    curriculum_type: str = "linear"  # linear, cosine, exponential


@dataclass
class DataConfig:
    processed_file: str = "data/processed_dataset.jsonl"
    annotated_file: str = "data/annotated_results.jsonl"
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    shuffle_seed: int = 42
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    max_total_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 50  # 只用50条做验证


@dataclass
class RewardConfig:
    # 课程式权重
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    # 四路奖励权重
    weight_diagnosis:    float = 0.4
    weight_completeness: float = 0.25
    weight_faithfulness: float = 0.2
    weight_format:       float = 0.15

    # LLM 温度控制
    llm_temperature: float = 0.3
    llm_temperature_decay: bool = True
    llm_temperature_min: float = 0.1
    llm_temperature_decay_steps: int = 500

    # 本地 LLM API（两路 LLM 奖励共用）
    api_base_url: str = "http://127.0.0.1:8080/v1"
    api_model:    str = "Qwen3.5-2B"
    api_timeout:     int = 60
    api_max_retries: int = 3

    reward_worker_threads: int = 4


@dataclass
class TrainConfig:
    output_dir: str = "./output/grpo_medical_v2"
    resume_from_checkpoint: Optional[str] = None
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    save_steps: int = 100
    do_eval: bool = True
    eval_strategy: str = 'steps'
    eval_steps: int = 100
    logging_steps: int = 10
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 2
    report_to: str = "tensorboard"
    seed: int = 42
    # GRPO 专有
    num_generations: int = 8
    num_generations_eval: int = 2        # eval 时少生成几条，节省显存
    temperature: float = 0.9
    top_p: float = 0.95
    epsilon: float = 0.2
    beta: float = 0.04
    loss_type: str = "grpo"
    # 原生样本打印（TRL 内置）
    log_completions: bool = True
    num_completions_to_print: int = 2
    gradient_checkpointing: bool = True                  # ← 加这里
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )


# 课程式权重调度器
class CurriculumScheduler:
    """根据训练步数动态调整四路奖励权重"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_step = 0
        self.init_weights = {
            "diagnosis":    config.init_weight_diagnosis,
            "completeness": config.init_weight_completeness,
            "faithfulness": config.init_weight_faithfulness,
            "format":       config.init_weight_format,
        }
        self.target_weights = {
            "diagnosis":    config.target_weight_diagnosis,
            "completeness": config.target_weight_completeness,
            "faithfulness": config.target_weight_faithfulness,
            "format":       config.target_weight_format,
        }
    
    def step(self, step: int = None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
    
    def get_weights(self, step: int = None) -> Dict[str, float]:
        if step is not None:
            self.current_step = step
        if not self.config.enable or self.current_step >= self.config.warmup_steps:
            return self.target_weights.copy()
        
        progress = self.current_step / max(1, self.config.warmup_steps)
        if self.config.curriculum_type == "linear":
            alpha = progress
        elif self.config.curriculum_type == "cosine":
            import math
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
        elif self.config.curriculum_type == "exponential":
            alpha = 1 - (1 - progress) ** 3
        else:
            alpha = progress
        
        current_weights = {}
        for key in self.init_weights:
            w_init = self.init_weights[key]
            w_target = self.target_weights[key]
            current_weights[key] = w_init + alpha * (w_target - w_init)
        return current_weights


# LLM 评分温度调度器
class TemperatureScheduler:
    def __init__(self, config: RewardConfig):
        self.config = config
        self.current_step = 0
    
    def step(self, step: int = None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
    
    def get_temperature(self, step: int = None) -> float:
        if step is not None:
            self.current_step = step
        if not self.config.llm_temperature_decay:
            return self.config.llm_temperature
        progress = min(1.0, self.current_step / max(1, self.config.llm_temperature_decay_steps))
        temp = self.config.llm_temperature - progress * (
            self.config.llm_temperature - self.config.llm_temperature_min
        )
        return max(self.config.llm_temperature_min, temp)


# ═══════════════════════════════════════════════════════════
#  2. Prompt 模板
# ═══════════════════════════════════════════════════════════

def build_prompt(medical_record: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"病历：\n{medical_record}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )


# ═══════════════════════════════════════════════════════════
#  3. 输出解析
# ═══════════════════════════════════════════════════════════

def parse_output(text: str) -> list[dict]:
    """返回 [{"ee_list": [...], "ec": "..."}, ...]"""
    results = []
    for block in re.findall(r"<DIAGNOSIS>(.*?)</DIAGNOSIS>", text, re.DOTALL):
        ee_list = [e.strip() for e in re.findall(r"<EE>(.*?)</EE>", block, re.DOTALL)]
        ec_m    = re.search(r"<EC>(.*?)</EC>", block, re.DOTALL)
        ec      = ec_m.group(1).strip() if ec_m else ""
        if ee_list or ec:
            results.append({"ee_list": ee_list, "ec": ec})
    return results


# ═══════════════════════════════════════════════════════════
#  4. 奖励计算器
#     四路奖励拆成四个独立函数，命名后传给 reward_funcs 列表，
#     TRL 自动按函数名记录 rewards/<name>/mean 和 rewards/<name>/std
# ═══════════════════════════════════════════════════════════

class RewardCalculator:
    def __init__(self, cfg: RewardConfig):
        self.cfg      = cfg
        self.executor = ThreadPoolExecutor(max_workers=cfg.reward_worker_threads)

        # 调度器
        self.curriculum_scheduler = CurriculumScheduler(cfg.curriculum)
        self.temperature_scheduler = TemperatureScheduler(cfg)
        
        # 新增：当前权重（会在训练中动态更新）
        self.current_weights = {
            "diagnosis":    cfg.weight_diagnosis,
            "completeness": cfg.weight_completeness,
            "faithfulness": cfg.weight_faithfulness,
            "format":       cfg.weight_format,
        }
        self.global_step = 0
        # 奖励缓存
        self._step_cache: tuple = (None, None)   # (completions_id, results_list)
        # OpenAI 客户端（本地部署）
        self.client = OpenAI(
            base_url=self.cfg.api_base_url,
            api_key="xxx"
        )

    # 新增：更新权重方法
    def update_weights(self, step: int):
        self.global_step = step
        if self.cfg.curriculum.enable:
            self.current_weights = self.curriculum_scheduler.get_weights(step)
            logger.info(
                f"Step {step}: 动态权重 → "
                f"诊断={self.current_weights['diagnosis']:.3f}, "
                f"完备性={self.current_weights['completeness']:.3f}, "
                f"忠实性={self.current_weights['faithfulness']:.3f}, "
                f"格式={self.current_weights['format']:.3f}"
            )
        self.temperature_scheduler.step(step)

    # ── 公共 LLM 调用 ────────────────────────────────────────
    def _call_llm(self, messages: list[dict], max_tokens: int, label: str) -> str:
        # 使用动态温度
        current_temp = self.temperature_scheduler.get_temperature()
        
        for attempt in range(self.cfg.api_max_retries):
            try:
                chat_response = self.client.chat.completions.create(
                    model=self.cfg.api_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=current_temp,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }, 
                )
                return chat_response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"[{label}] 第{attempt+1}次失败 (温度={current_temp:.2f}): {e}")
        logger.error(f"[{label}] 达到最大重试次数")
        return ""

    # ── 诊断准确性 ───────────────────────────────────────────
    def _diagnosis_accuracy(self, parsed: list[dict], true_diagnoses: list[str]) -> float:
        if not parsed:
            return 0.0
        ee = "\n".join("\n".join(d["ee_list"]) for d in parsed if d["ee_list"])
        ec = "\n".join(d["ec"] for d in parsed if d["ec"])
        if not ee.strip():
            return 0.0
        raw = self._call_llm(
            [{"role": "system", "content": _DIAG_SYS},
             {"role": "user",   "content": f"证据元素：\n{ee}\n\n证据链：\n{ec}"}],
            max_tokens=1024, label="diagnosis",
        )
        if not raw:
            return 0.0
        predicted = {l.strip().lower() for l in raw.splitlines() if l.strip()}
        true_set  = {d.strip().lower() for d in true_diagnoses if d.strip()}
        if not true_set:
            return 0.0
        hit = sum(1 for t in true_set if any(t in p or p in t for p in predicted))
        return hit / len(true_set)

    # ── 证据完备性 ───────────────────────────────────────────
    _SCORE_RE = re.compile(r"\b(1\.0|0\.5|0\.0|1|0)\b")

    def _parse_score(self, raw: str) -> float:
        try:
            v = float(raw.strip())
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
        m = self._SCORE_RE.search(raw)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        low = raw.lower()
        if any(k in low for k in ("完整", "完全", "1.0", "full")):   return 1.0
        if any(k in low for k in ("部分", "0.5", "partial")):        return 0.5
        if any(k in low for k in ("缺失", "0.0", "missing")):        return 0.0
        return 0.0

    def _completeness(self, medical_record: str, parsed: list[dict],
                      true_diagnoses: list[str]) -> float:
        if not parsed:
            return 0.0
        ee_summary   = "\n".join("\n".join(f"  - {e}" for e in d["ee_list"]) for d in parsed)
        true_diag_str = "、".join(true_diagnoses) if true_diagnoses else "（未知）"
        user = (f"【病历原文】\n{medical_record}\n\n"
                f"【真实诊断列表】\n{true_diag_str}\n\n"
                f"【待评估证据元素】\n{ee_summary}\n\n"
                f"请只输出一个数字：1.0 或 0.5 或 0.0")
        raw = self._call_llm(
            [{"role": "system", "content": _COMP_SYS},
             {"role": "user",   "content": user}],
            max_tokens=1024, label="completeness",
        )
        return self._parse_score(raw)

    # ── 证据忠实性（规则）───────────────────────────────────
    def _faithfulness(self, medical_record: str, parsed: list[dict]) -> float:
        all_ee = [e for d in parsed for e in d["ee_list"]]
        if not all_ee:
            return 0.0
        for ee in all_ee:
            if ee.strip() and ee.strip() not in medical_record:
                logger.debug(f"[忠实性] EE 不在病历: {ee[:60]!r}")
                return 0.0
        return 1.0

    # ── 格式合规性（规则）───────────────────────────────────
    def _format(self, generation: str, parsed: list[dict]) -> float:
        if re.search(r"<n>.*?</n>", generation, re.DOTALL):
            return 0.0
        blocks = re.findall(r"<DIAGNOSIS>(.*?)</DIAGNOSIS>", generation, re.DOTALL)
        if not blocks:
            return 0.0
        perfect = sum(
            1 for b in blocks
            if re.search(r"<EE>.+?</EE>", b, re.DOTALL)
            and re.search(r"<EC>.+?</EC>", b, re.DOTALL)
        )
        partial = sum(
            1 for b in blocks
            if (re.search(r"<EE>.+?</EE>", b, re.DOTALL)
                or re.search(r"<EC>.+?</EC>", b, re.DOTALL))
            and not (re.search(r"<EE>.+?</EE>", b, re.DOTALL)
                     and re.search(r"<EC>.+?</EC>", b, re.DOTALL))
        )
        if perfect == len(blocks):       return 1.0
        if perfect + partial == len(blocks): return 0.5
        return 0.0

    # ── 单条总奖励 ───────────────────────────────────────────
    def _compute_one(self, generation: str, medical_record: str,
                     true_diagnoses: list[str]) -> dict:
        parsed  = parse_output(generation)
        r_diag  = self._diagnosis_accuracy(parsed, true_diagnoses)
        r_comp  = self._completeness(medical_record, parsed, true_diagnoses)
        r_faith = self._faithfulness(medical_record, parsed)
        r_fmt   = self._format(generation, parsed)
        return {
            "diagnosis":    r_diag,
            "completeness": r_comp,
            "faithfulness": r_faith,
            "format":       r_fmt,
        }

    def _batch(self, completions, medical_records, true_diagnoses_json):
        def _parse(s):
            try:    return json.loads(s)
            except: return [s] if s else []

        futures = [
            self.executor.submit(self._compute_one, gen, rec, _parse(d))
            for gen, rec, d in zip(completions, medical_records, true_diagnoses_json)
        ]
        results = []
        for f in futures:
            try:    results.append(f.result(timeout=120))
            except Exception as e:
                logger.error(f"奖励计算异常: {e}")
                results.append({"diagnosis": 0.0, "completeness": 0.0,
                                "faithfulness": 0.0, "format": 0.0})
        return results
    
    # ✅ 新增：带缓存的批量计算，替换四个函数里的直接 _batch 调用
    def _batch_once(self, completions, medical_records, true_diagnoses_json):
        """同一step内四路奖励共享一次 _batch 调用"""
        cache_key = id(completions)   # 同一step内同一个list对象，id不变
        if self._step_cache[0] == cache_key:
            return self._step_cache[1]
        
        results = self._batch(completions, medical_records, true_diagnoses_json)
        self._step_cache = (cache_key, results)   # 覆盖上一step的缓存
        return results

    # ── 四个独立奖励函数（按函数名自动记录到日志）────────────
    def reward_diagnosis(self, completions, **kwargs):
        recs   = kwargs.get("medical_record",  [""] * len(completions))
        diags  = kwargs.get("true_diagnoses",  ["[]"] * len(completions))
        weight = self.current_weights["diagnosis"]
        return [r["diagnosis"] * weight
                for r in self._batch_once(completions, recs, diags)]

    def reward_completeness(self, completions, **kwargs):
        recs   = kwargs.get("medical_record",  [""] * len(completions))
        diags  = kwargs.get("true_diagnoses",  ["[]"] * len(completions))
        weight = self.current_weights["completeness"]
        return [r["completeness"] * weight
                for r in self._batch_once(completions, recs, diags)]

    def reward_faithfulness(self, completions, **kwargs):
        recs   = kwargs.get("medical_record",  [""] * len(completions))
        diags  = kwargs.get("true_diagnoses",  ["[]"] * len(completions))
        weight = self.current_weights["faithfulness"]
        return [r["faithfulness"] * weight
                for r in self._batch_once(completions, recs, diags)]

    def reward_format(self, completions, **kwargs):
        recs   = kwargs.get("medical_record",  [""] * len(completions))
        diags  = kwargs.get("true_diagnoses",  ["[]"] * len(completions))
        weight = self.current_weights["format"]
        return [r["format"] * weight
                for r in self._batch_once(completions, recs, diags)]


class CurriculumWeightCallback(TrainerCallback):
    """在每个步数更新奖励权重"""
    def __init__(self, reward_calculator: RewardCalculator):
        self.calc = reward_calculator
    
    def on_step_end(self, args, state, control, **kwargs):
        self.calc.update_weights(state.global_step)
        return control

# ═══════════════════════════════════════════════════════════
#  5. 数据集构建
# ═══════════════════════════════════════════════════════════

def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def merge_and_split(cfg: DataConfig, tokenizer) -> tuple[Dataset, Dataset, Dataset]:
    logger.info(f"加载 processed_dataset: {cfg.processed_file}")
    processed = {str(r["患者id"]): r for r in _load_jsonl(cfg.processed_file)}
    logger.info(f"加载 annotated_results: {cfg.annotated_file}")
    annotated = {str(r["患者id"]): r for r in _load_jsonl(cfg.annotated_file)}

    common_ids = sorted(set(processed) & set(annotated))
    logger.info(f"交集患者数: {len(common_ids)}")
    if cfg.max_total_samples:
        common_ids = common_ids[:cfg.max_total_samples]

    rng = random.Random(cfg.shuffle_seed)
    rng.shuffle(common_ids)
    n       = len(common_ids)
    n_train = int(n * cfg.train_ratio)
    n_eval  = int(n * cfg.eval_ratio)
    splits  = {
        "train": common_ids[:n_train],
        "eval":  common_ids[n_train: n_train + n_eval],
        "test":  common_ids[n_train + n_eval:],
    }
    # 截取最大的 eval 样本数，避免 eval 太大时频繁调用 LLM 评分导致训练过慢
    splits["eval"] = splits["eval"][:cfg.max_eval_samples] if cfg.max_eval_samples else splits["eval"]
    logger.info(f"划分 → train:{len(splits['train'])} eval:{len(splits['eval'])} test:{len(splits['test'])}")

    def _build(ids: list[str]) -> Dataset:
        prompts, records, diags, refs = [], [], [], []
        for pid in ids:
            proc        = processed[pid]
            ann         = annotated[pid]
            med_record  = proc.get("input", "")
            diag_entries: list[dict] = ann.get("诊断列表", [])
            true_names  = [d.get("诊断名称", "") for d in diag_entries]

            ref_parts = []
            for d in diag_entries:
                ee_tags = "\n".join(f"<EE>{e}</EE>" for e in d.get("证据元素列表", []))
                ec_tag  = f"<EC>{d.get('证据链', '')}</EC>"
                ref_parts.append(f"<DIAGNOSIS>\n{ee_tags}\n{ec_tag}\n</DIAGNOSIS>")

            prompts.append(build_prompt(med_record, tokenizer))
            records.append(med_record)
            diags.append(json.dumps(true_names, ensure_ascii=False))
            refs.append("\n\n".join(ref_parts))

        return Dataset.from_dict({
            "prompt":         prompts,
            "medical_record": records,
            "true_diagnoses": diags,
            "ref_output":     refs,
        })

    return _build(splits["train"]), _build(splits["eval"]), _build(splits["test"])


# ═══════════════════════════════════════════════════════════
#  6. 主训练入口
# ═══════════════════════════════════════════════════════════

def train(model_cfg: ModelConfig, data_cfg: DataConfig,
          reward_cfg: RewardConfig, train_cfg: TrainConfig):

    logger.info(f"加载 tokenizer: {model_cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path, trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds, test_ds = merge_and_split(data_cfg, tokenizer)
    test_save = os.path.join(train_cfg.output_dir, "test_dataset")
    os.makedirs(test_save, exist_ok=True)
    test_ds.save_to_disk(test_save)
    logger.info(f"测试集已保存至 {test_save}（共 {len(test_ds)} 条）")

    calc = RewardCalculator(reward_cfg)

    grpo_config = GRPOConfig(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        max_grad_norm=train_cfg.max_grad_norm,
        save_steps=train_cfg.save_steps,
        do_eval=True,
        eval_strategy='steps',
        eval_steps=train_cfg.eval_steps,
        logging_steps=train_cfg.logging_steps,
        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        report_to=train_cfg.report_to,
        seed=train_cfg.seed,
        num_generations=train_cfg.num_generations,
        num_generations_eval=train_cfg.num_generations_eval,
        temperature=train_cfg.temperature,
        top_p=train_cfg.top_p,
        epsilon=train_cfg.epsilon,
        beta=train_cfg.beta,
        loss_type=train_cfg.loss_type,
        # max_prompt_length=data_cfg.max_prompt_length,
        max_completion_length=data_cfg.max_completion_length,
        generation_kwargs={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        # 原生打印生成样本
        log_completions=train_cfg.log_completions,
        num_completions_to_print=train_cfg.num_completions_to_print,
        # device_map=None 避免与多卡 DDP 冲突
        model_init_kwargs={
            "trust_remote_code":   model_cfg.trust_remote_code,
            "torch_dtype":         model_cfg.torch_dtype,
            # "attn_implementation": model_cfg.attn_implementation,
            "device_map":          None,
        },
        use_liger_kernel=False,
        ds3_gather_for_generation=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 与 flash_attention_2 兼容
    )

    actual_model_path = model_cfg.model_name_or_path
    if train_cfg.resume_from_checkpoint and os.path.isdir(train_cfg.resume_from_checkpoint):
        # checkpoint 目录里有完整模型权重，直接用它初始化
        actual_model_path = train_cfg.resume_from_checkpoint
        logger.info(f"从 checkpoint 恢复，模型路径覆盖为: {actual_model_path}")
    elif train_cfg.resume_from_checkpoint:
        logger.warning(f"checkpoint 路径不存在: {train_cfg.resume_from_checkpoint}，使用原始模型")
        train_cfg.resume_from_checkpoint = None

    trainer = GRPOTrainer(
        model=actual_model_path,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # 四个命名函数，TRL 自动记录 rewards/reward_diagnosis/mean 等
        reward_funcs=[
            calc.reward_diagnosis,
            calc.reward_completeness,
            calc.reward_faithfulness,
            calc.reward_format,
        ],
        processing_class=tokenizer,
        callbacks=[CurriculumWeightCallback(calc)],
    )

    logger.info("开始 GRPO 训练...")
    logger.info(f"课程学习: {'启用' if reward_cfg.curriculum.enable else '禁用'}")
    logger.info(f"LLM评分温度: {reward_cfg.llm_temperature}")

    if train_cfg.resume_from_checkpoint == "":
        train_cfg.resume_from_checkpoint = None
    logger.info(f"复用checkpoint: {train_cfg.resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=train_cfg.resume_from_checkpoint)

    final_dir = os.path.join(train_cfg.output_dir, "final")
    logger.info(f"保存最终模型至 {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("训练完成！")


# ═══════════════════════════════════════════════════════════
#  7. CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="GRPO Medical v2")

    g = p.add_argument_group("Model")
    g.add_argument("--model_name_or_path", default="/data_net/models_for_all/Qwen2.5-0.5B")
    g.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    g.add_argument("--attn_implementation", default="flash_attention_2",
                   choices=["eager","sdpa","flash_attention_2"])

    g = p.add_argument_group("Data")
    g.add_argument("--processed_file", default="data/processed_dataset.jsonl")
    g.add_argument("--annotated_file", default="data/annotated_results.jsonl")
    g.add_argument("--train_ratio",  type=float, default=0.8)
    g.add_argument("--eval_ratio",   type=float, default=0.1)
    g.add_argument("--shuffle_seed", type=int,   default=42)
    g.add_argument("--max_prompt_length",     type=int, default=2048)
    g.add_argument("--max_completion_length", type=int, default=2048)
    g.add_argument("--max_total_samples",     type=int, default=None)
    g.add_argument("--max_eval_samples",     type=int, default=None)

    g = p.add_argument_group("Reward")
    g.add_argument("--weight_diagnosis",    type=float, default=0.4)
    g.add_argument("--weight_completeness", type=float, default=0.25)
    g.add_argument("--weight_faithfulness", type=float, default=0.2)
    g.add_argument("--weight_format",       type=float, default=0.15)
    g.add_argument("--llm_temperature", type=float, default=0.3)
    g.add_argument("--llm_temperature_decay", action="store_true", default=True)
    g.add_argument("--llm_temperature_min", type=float, default=0.1)
    g.add_argument("--llm_temperature_decay_steps", type=int, default=500)
    g.add_argument("--api_base_url", default="http://127.0.0.1:8080/v1")
    g.add_argument("--api_model",    default="Qwen3.5-2B")
    g.add_argument("--api_timeout",     type=int, default=60)
    g.add_argument("--api_max_retries", type=int, default=3)
    g.add_argument("--reward_worker_threads", type=int, default=4)

    g = p.add_argument_group("Training")
    g.add_argument("--resume_from_checkpoint", type=str, default=None,
               help="从指定 checkpoint 目录恢复训练，如 ./output/checkpoint-300")
    g.add_argument("--output_dir", default="./output/grpo_medical_v2")
    g.add_argument("--num_train_epochs",            type=int,   default=3)
    g.add_argument("--per_device_train_batch_size", type=int,   default=2)
    g.add_argument("--gradient_accumulation_steps", type=int,   default=8)
    g.add_argument("--learning_rate",               type=float, default=1e-6)
    g.add_argument("--warmup_ratio",                type=float, default=0.05)
    g.add_argument("--max_grad_norm",               type=float, default=1.0)
    g.add_argument("--save_steps",    type=int, default=100)
    g.add_argument("--do_eval", action='store_true', default=True)
    g.add_argument('--eval_strategy', type=str, default='steps')
    g.add_argument("--eval_steps",    type=int, default=100)
    g.add_argument("--logging_steps", type=int, default=10)
    g.add_argument("--report_to", default="tensorboard",
                   choices=["none","tensorboard","wandb"])
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--log_completions",       action="store_true", default=True)
    g.add_argument("--num_completions_to_print", type=int, default=2)

    g = p.add_argument_group("Curriculum")
    g.add_argument("--curriculum_enable", action="store_true", default=True)
    g.add_argument("--curriculum_warmup_steps", type=int, default=200)
    g.add_argument("--curriculum_type", default="linear",
                   choices=["linear", "cosine", "exponential"])
    g.add_argument("--init_weight_diagnosis",    type=float, default=0.1)
    g.add_argument("--init_weight_completeness", type=float, default=0.1)
    g.add_argument("--init_weight_faithfulness", type=float, default=0.4)
    g.add_argument("--init_weight_format",       type=float, default=0.4)
    g.add_argument("--target_weight_diagnosis",    type=float, default=0.4)
    g.add_argument("--target_weight_completeness", type=float, default=0.25)
    g.add_argument("--target_weight_faithfulness", type=float, default=0.2)
    g.add_argument("--target_weight_format",       type=float, default=0.15)

    g = p.add_argument_group("GRPO")
    g.add_argument("--num_generations",      type=int,   default=8)
    g.add_argument("--num_generations_eval", type=int,   default=2)
    g.add_argument("--temperature",          type=float, default=0.9)
    g.add_argument("--top_p",                type=float, default=0.95)
    g.add_argument("--epsilon",              type=float, default=0.2)
    g.add_argument("--beta",                 type=float, default=0.04)
    g.add_argument("--loss_type", default="grpo", choices=["grpo","bnpo","dr_grpo"])

    return p.parse_args()


def main():
    args = parse_args()
    train(
        ModelConfig(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
        ),
        DataConfig(
            processed_file=args.processed_file,
            annotated_file=args.annotated_file,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            shuffle_seed=args.shuffle_seed,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            max_total_samples=args.max_total_samples,
            max_eval_samples=args.max_eval_samples
        ),
        RewardConfig(
            curriculum=CurriculumConfig(
                enable=args.curriculum_enable,
                init_weight_diagnosis=args.init_weight_diagnosis,
                init_weight_completeness=args.init_weight_completeness,
                init_weight_faithfulness=args.init_weight_faithfulness,
                init_weight_format=args.init_weight_format,
                target_weight_diagnosis=args.target_weight_diagnosis,
                target_weight_completeness=args.target_weight_completeness,
                target_weight_faithfulness=args.target_weight_faithfulness,
                target_weight_format=args.target_weight_format,
                warmup_steps=args.curriculum_warmup_steps,
                curriculum_type=args.curriculum_type,
            ),
            weight_diagnosis=args.weight_diagnosis,
            weight_completeness=args.weight_completeness,
            weight_faithfulness=args.weight_faithfulness,
            weight_format=args.weight_format,
            llm_temperature=args.llm_temperature,
            llm_temperature_decay=args.llm_temperature_decay,
            llm_temperature_min=args.llm_temperature_min,
            llm_temperature_decay_steps=args.llm_temperature_decay_steps,
            api_base_url=args.api_base_url,
            api_model=args.api_model,
            api_timeout=args.api_timeout,
            api_max_retries=args.api_max_retries,
            reward_worker_threads=args.reward_worker_threads,
        ),
        TrainConfig(
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            save_steps=args.save_steps,
            do_eval=args.do_eval,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            report_to=args.report_to,
            seed=args.seed,
            num_generations=args.num_generations,
            num_generations_eval=args.num_generations_eval,
            temperature=args.temperature,
            top_p=args.top_p,
            epsilon=args.epsilon,
            beta=args.beta,
            loss_type=args.loss_type,
            log_completions=args.log_completions,
            num_completions_to_print=args.num_completions_to_print,
        ),
    )


if __name__ == "__main__":
    main()