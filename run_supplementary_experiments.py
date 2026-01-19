import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from experiment_logger import ExperimentLogger
from main import run_ai_column_generation, generate_ships, Ship, TOTAL_STEPS, N_SP, K_TOTAL


# ==========================================
# 1. 强力 Baseline: Time-Limited MIP (限制时间的求解器)
# ==========================================
def run_mip_heuristic(
    n_ships=100,
    time_limit=15.0,
    seed=42,
    logger=None,
    instance_id=None,
    scenario="heterogeneous",
):
    """
    直接把大问题扔给 Gurobi，限制它只能跑 time_limit 秒。
    这是工业界最常用的方法，也是最有力的对比对象。
    """
    # 使用 main.py 里同样的生成逻辑 (需确保 main.py 已注入异质性补丁，或者直接在此重新定义)
    # 为了保险，我们这里重新生成同样配置的数据
    ships = generate_ships(n=n_ships, cost_battery_val=120.0, seed=seed)

    # 重新注入异质性 (模拟 run_comparison_final.py 的逻辑)
    rng = np.random.default_rng(seed)
    for s in ships:
        s.cost_battery = float(rng.integers(60, 250))

    print(f">>> 正在运行 Time-Limited Exact CG (Limit={time_limit}s)...")
    result = run_ai_column_generation(
        n_ships=n_ships,
        enable_ai=False,
        battery_cost=120.0,
        n_sp=N_SP,
        seed=seed,
        time_limit=time_limit,
    )

    result_payload = {
        "Method": f"Exact CG (Time-Limited {time_limit}s)",
        "Time": result["time"],
        "Cost": result["obj"],
    }
    if logger is not None:
        logger.log_row(
            {
                "instance_id": instance_id or f"N{n_ships}_seed{seed}_{scenario}",
                "N": n_ships,
                "seed": seed,
                "scenario": scenario,
                "method": result_payload["Method"],
                "obj": result_payload["Cost"],
                "runtime_total": result_payload["Time"],
                "runtime_rmp": result.get("runtime_rmp", 0.0),
                "runtime_pricing": result.get("runtime_pricing", 0.0),
                "status": "ok",
                "gap": None,
                "num_iters": result.get("iter"),
                "num_pricing_calls": result.get("pricing_calls", 0),
                "num_fallback_calls": result.get("fallback_calls", 0),
                "num_columns_added": result.get("columns_added", 0),
                "min_reduced_cost_last": result.get("min_reduced_cost_last"),
                "pricing_time_share": result.get("pricing_time_share", 0.0),
            }
        )
    return result_payload


# ==========================================
# 2. 消融实验: MLP vs GNN (证明图结构有用)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        # 比 GNN 简单的网络，层数少，参数少
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def run_ablation_study(csv_path="ablation_accuracy.csv"):
    print("\n>>> [消融实验] 正在对比 GNN (Proposed) vs. Simple MLP (Baseline)...")
    if not pd.io.common.file_exists(csv_path):
        print(f"⚠️ 缺少消融准确率数据: {csv_path}，请先生成真实结果。")
        return

    df = pd.read_csv(csv_path)
    if not {"Method", "Accuracy"}.issubset(df.columns):
        raise ValueError("ablation_accuracy.csv 必须包含 Method 和 Accuracy 列。")

    methods = df["Method"].tolist()
    accuracies = df["Accuracy"].tolist()
    colors = [None] * len(methods)

    # 画图
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')

    # 添加纹理
    patterns = ['//', '', '..'] + [''] * max(0, len(bars) - 3)
    for bar, pat in zip(bars, patterns):
        bar.set_hatch(pat)

    plt.ylim(70, 100)
    plt.ylabel("Prediction Accuracy (%)", fontweight='bold')
    plt.title("Ablation Study: Impact of Model Architecture", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 标数值
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{bar.get_height()}%", ha='center', fontweight='bold')

    plt.savefig("Fig_Ablation_Model.png", dpi=300)
    print("✅ 消融实验图已生成: Fig_Ablation_Model.png")


# ==========================================
# 3. 更新帕累托图 (加入 MIP Heuristic)
# ==========================================
def update_pareto_with_mip():
    try:
        # 读取旧数据
        df = pd.read_csv("exp_final_comparison.csv")
        subset = df[df['Scale'] == 100].copy()

        # 绘图
        plt.figure(figsize=(9, 6))

        # 定义样式
        styles = {
            'Greedy Heuristic': {'c': '#999999', 'm': 'x', 's': 300},
            'Exact CG': {'c': '#2ca02c', 'm': 's', 's': 300},
            'GNN-Accelerated CG': {'c': '#d62728', 'm': '*', 's': 500},  # 我们的最显眼
            'Exact CG (Time-Limited 15.0s)': {'c': '#1f77b4', 'm': '^', 's': 300}
        }

        # 画旧点
        for method in subset['Method'].unique():
            row = subset[subset['Method'] == method].iloc[0]
            st = styles.get(method, {'c': 'k', 'm': 'o', 's': 100})
            plt.scatter(row['Time'], row['Cost'], s=st['s'], c=st['c'], marker=st['m'], label=method)
            plt.text(row['Time'], row['Cost'] + 150, f"{int(row['Cost'])}", ha='center', fontsize=10)

        # 装饰
        plt.xlabel("Computation Time (s)", fontweight='bold')
        plt.ylabel("Total Cost ($)", fontweight='bold')
        plt.title("Algorithm Performance: GNN vs. Traditional Methods", fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.savefig("Fig_Pareto_Enhanced.png", dpi=300)
        print("✅ 增强版帕累托图已生成: Fig_Pareto_Enhanced.png")

    except Exception as e:
        print(f"绘图失败: {e}")


if __name__ == "__main__":
    logger = ExperimentLogger("results.csv")
    run_mip_heuristic(logger=logger)
    run_ablation_study()
    update_pareto_with_mip()
