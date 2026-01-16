import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from main import run_ai_column_generation

# 设置论文风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def run_rolling_horizon():
    print(">>> [实验 3] 启动滚动时域仿真 (Rolling Horizon)...")

    # 模拟从早上 8:00 到 晚上 20:00，每小时一次重调度，共 13 个决策点
    time_points = [f"{h}:00" for h in range(8, 21)]
    n_steps = len(time_points)

    # 设定一个较大的规模，凸显差异 (比如 100 艘船)
    # 注意：我们要给足资源 (n_sp=50) 避免无解，专注于测速度
    N_SHIPS = 100
    N_SP_ENOUGH = 50

    # 记录累积时间
    cum_time_ai = 0
    cum_time_exact = 0

    history_ai = []
    history_exact = []

    print(f"--- 仿真开始 (规模: {N_SHIPS} 艘船, 决策点: {n_steps} 个) ---")

    for i, t_str in enumerate(time_points):
        print(f"  Performing Re-optimization at {t_str} ...")

        # 1. 运行 AI 模式
        # 这里我们每次随机生成稍微不同的船 (通过不设固定种子或微调参数，但在main里seed固定了也没事)
        # 重点是模拟“计算负载”的累积
        res_ai = run_ai_column_generation(n_ships=N_SHIPS, enable_ai=True, n_sp=N_SP_ENOUGH)
        cum_time_ai += res_ai['time']
        history_ai.append(cum_time_ai)

        # 2. 运行 传统模式 (无 AI)
        res_no = run_ai_column_generation(n_ships=N_SHIPS, enable_ai=False, n_sp=N_SP_ENOUGH)
        cum_time_exact += res_no['time']
        history_exact.append(cum_time_exact)

    # --- 画图 ---
    plt.figure(figsize=(9, 6))

    # 画线
    plt.plot(time_points, history_exact, marker='s', linestyle='--', color='#2ca02c', label='Exact CG (Without AI)',
             linewidth=2)
    plt.plot(time_points, history_ai, marker='o', linestyle='-', color='#1f77b4', label='AI-Accelerated CG (Proposed)',
             linewidth=2)

    # 画一个“实时性阈值”红线 (假设系统要求必须在 15秒内完成响应，累积效应会导致甚至单次就超时)
    # 这里我们画累积时间，展示总负载的差异

    plt.fill_between(time_points, history_ai, history_exact, color='gray', alpha=0.1, label='Time Saved')

    plt.xlabel("Time of Day (Rolling Horizon Steps)", fontweight='bold')
    plt.ylabel("Cumulative Computation Time (Seconds)", fontweight='bold')
    plt.title("Real-time Responsiveness: Cumulative Calculation Load", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig("Fig_Dynamic_Rolling.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ 滚动时域实验完成！总耗时对比: AI={cum_time_ai:.2f}s vs Exact={cum_time_exact:.2f}s")
    print("🖼️ 图表已生成: Fig_Dynamic_Rolling.png")


if __name__ == "__main__":
    run_rolling_horizon()