import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 配置区
# =========================
INPUT_JSON = Path("/data1/med_group/grpo_trl/grpo_medical_v2/checkpoint-8325/trainer_state.json")
OUT_DIR = Path("output/grpo_training_curves")
ROLLING_WINDOW = 25
# 只保留 step <= STEP_CUTOFF 的数据点
STEP_CUTOFF = 8310

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 数据读取与拆分
# =========================
with INPUT_JSON.open("r", encoding="utf-8") as f:
    state = json.load(f)

log_history = state.get("log_history", [])
df = pd.DataFrame(log_history)

if df.empty:
    raise ValueError("trainer_state.json 中没有 log_history，无法绘图。")

if "step" not in df.columns:
    raise ValueError("log_history 中缺少 step 字段，无法以 step 作为横轴。")

df["step"] = pd.to_numeric(df["step"], errors="coerce")

# 训练日志：含 reward 的行
train_df = df[df.get("reward").notna()].copy() if "reward" in df.columns else pd.DataFrame()
train_df = train_df[train_df["step"] <= STEP_CUTOFF].copy()

# 评估日志：含 eval_reward 的行
eval_df = df[df.get("eval_reward").notna()].copy() if "eval_reward" in df.columns else pd.DataFrame()
eval_df = eval_df[eval_df["step"] <= STEP_CUTOFF].copy()

for data in (train_df, eval_df):
    if not data.empty:
        data.sort_values("step", inplace=True)


# =========================
# 3. 通用绘图函数
# =========================
def plot_metric(
    filename: str,
    title: str,
    ylabel: str,
    series: list[tuple[pd.DataFrame, str, str]],
    rolling_window: int | None = ROLLING_WINDOW,
    show_raw: bool = True,
    marker_for_eval: bool = True,
):
    """
    绘制单指标或同类指标曲线。

    Args:
        filename:
            输出图片文件名。
        title:
            图标题。
        ylabel:
            Y 轴名称。
        series:
            列表，每项格式为：
            (dataframe, column_name, label)
        rolling_window:
            平滑窗口。为 None 时不绘制 rolling 曲线。
        show_raw:
            是否绘制原始曲线。
        marker_for_eval:
            eval 点通常较少，是否加 marker。
    """
    plt.figure(figsize=(11, 6))
    plotted = False

    for data, col, label in series:
        if data is None or data.empty or col not in data.columns:
            continue

        temp = data[["step", col]].dropna().copy()
        if temp.empty:
            continue

        x = temp["step"]
        y = temp[col]

        is_eval = col.startswith("eval_") or label.lower().startswith("eval")

        if show_raw:
            if is_eval and marker_for_eval:
                plt.plot(x, y, linewidth=1.5, marker="o", markersize=3, alpha=0.7, label=f"{label} raw")
            else:
                plt.plot(x, y, linewidth=1.0, alpha=0.35, label=f"{label} raw")

        if rolling_window is not None and len(temp) >= max(3, rolling_window // 3):
            smooth = y.rolling(
                rolling_window,
                min_periods=max(3, rolling_window // 3),
            ).mean()
            plt.plot(x, smooth, linewidth=2.2, label=f"{label} rolling-{rolling_window}")

        plotted = True

    if not plotted:
        plt.close()
        print(f"[skip] {filename}: 没有可绘制的数据列")
        return None

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = OUT_DIR / filename
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"[saved] {output_path}")
    return output_path


# =========================
# 4. 按新要求绘图
# =========================

# 01. 总 reward：train vs eval
plot_metric(
    filename="01_reward_train_eval.png",
    title="GRPO Reward Curve: Train vs Eval",
    ylabel="reward",
    series=[
        (train_df, "reward", "train reward"),
        (eval_df, "eval_reward", "eval reward"),
    ],
)

# 02. KL 单独绘制
plot_metric(
    filename="02_kl_train_eval.png",
    title="GRPO KL Curve: Train vs Eval",
    ylabel="kl",
    series=[
        (train_df, "kl", "train KL"),
        (eval_df, "eval_kl", "eval KL"),
    ],
)

# 03. Entropy 单独绘制
plot_metric(
    filename="03_entropy_train_eval.png",
    title="GRPO Entropy Curve: Train vs Eval",
    ylabel="entropy",
    series=[
        (train_df, "entropy", "train entropy"),
        (eval_df, "eval_entropy", "eval entropy"),
    ],
)

# 04. Loss 单独绘制
plot_metric(
    filename="04_loss_train_eval.png",
    title="GRPO Loss Curve: Train vs Eval",
    ylabel="loss",
    series=[
        (train_df, "loss", "train loss"),
        (eval_df, "eval_loss", "eval loss"),
    ],
)

# 05. Grad Norm 单独绘制
plot_metric(
    filename="05_grad_norm_train.png",
    title="GRPO Grad Norm Curve",
    ylabel="grad_norm",
    series=[
        (train_df, "grad_norm", "train grad_norm"),
    ],
)

# 06. Completion Length 单独绘制
plot_metric(
    filename="06_completion_length_train_eval.png",
    title="GRPO Completion Mean Length: Train vs Eval",
    ylabel="mean completion length",
    series=[
        (train_df, "completions/mean_length", "train mean length"),
        (eval_df, "eval_completions/mean_length", "eval mean length"),
    ],
)

# 07. Clipping 单独绘制
plot_metric(
    filename="07_clipping_ratio_train_eval.png",
    title="GRPO Completion Clipped Ratio: Train vs Eval",
    ylabel="clipped ratio",
    series=[
        (train_df, "completions/clipped_ratio", "train clipped ratio"),
        (eval_df, "eval_completions/clipped_ratio", "eval clipped ratio"),
    ],
)

# 08. Learning Rate
plot_metric(
    filename="08_learning_rate.png",
    title="Learning Rate Schedule",
    ylabel="learning rate",
    series=[
        (train_df, "learning_rate", "learning rate"),
    ],
    rolling_window=None,
    show_raw=True,
)


# =========================
# 5. 导出关键 CSV
# =========================
selected_train_cols = [
    "step",
    "epoch",
    "reward",
    "reward_std",
    "kl",
    "entropy",
    "loss",
    "grad_norm",
    "learning_rate",
    "completions/mean_length",
    "completions/mean_terminated_length",
    "completions/clipped_ratio",
    "num_tokens",
    "step_time",
]

selected_eval_cols = [
    "step",
    "epoch",
    "eval_reward",
    "eval_reward_std",
    "eval_kl",
    "eval_entropy",
    "eval_loss",
    "eval_completions/mean_length",
    "eval_completions/mean_terminated_length",
    "eval_completions/clipped_ratio",
    "eval_runtime",
]

train_export = train_df[[c for c in selected_train_cols if c in train_df.columns]].copy()
eval_export = eval_df[[c for c in selected_eval_cols if c in eval_df.columns]].copy()

train_export.to_csv(OUT_DIR / "train_selected_metrics.csv", index=False)
eval_export.to_csv(OUT_DIR / "eval_selected_metrics.csv", index=False)


# =========================
# 6. 导出 summary
# =========================
summary = {
    "global_step": state.get("global_step"),
    "epoch": state.get("epoch"),
    "eval_steps": state.get("eval_steps"),
    "train_rows": len(train_df),
    "eval_rows": len(eval_df),
}

if not train_df.empty and "reward" in train_df.columns:
    train_reward = train_df[["step", "reward"]].dropna()
    if not train_reward.empty:
        idx = train_reward["reward"].idxmax()
        summary.update(
            {
                "first_train_step": int(train_df["step"].min()),
                "last_train_step": int(train_df["step"].max()),
                "final_train_reward": float(train_reward["reward"].iloc[-1]),
                "max_train_reward": float(train_reward.loc[idx, "reward"]),
                "max_train_reward_step": int(train_reward.loc[idx, "step"]),
            }
        )

if not eval_df.empty and "eval_reward" in eval_df.columns:
    eval_reward = eval_df[["step", "eval_reward"]].dropna()
    if not eval_reward.empty:
        idx = eval_reward["eval_reward"].idxmax()
        summary.update(
            {
                "first_eval_step": int(eval_df["step"].min()),
                "last_eval_step": int(eval_df["step"].max()),
                "final_eval_reward": float(eval_reward["eval_reward"].iloc[-1]),
                "max_eval_reward": float(eval_reward.loc[idx, "eval_reward"]),
                "max_eval_reward_step": int(eval_reward.loc[idx, "step"]),
            }
        )

pd.DataFrame([summary]).to_csv(OUT_DIR / "summary_metrics.csv", index=False)

print("\nSummary:")
for k, v in summary.items():
    print(f"{k}: {v}")
