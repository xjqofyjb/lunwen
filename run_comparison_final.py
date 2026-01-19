import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import main  # 导入 main 模块以便进行“热修补”
from main import run_ai_column_generation, TOTAL_STEPS, N_SP
from experiment_logger import ExperimentLogger


# ==========================================
# 🔧 魔法补丁：定义一个带“贫富差距”的生成函数
# ==========================================
def generate_heterogeneous_ships(n=20, cost_battery_val=120.0, seed=42):
    """
    这个函数会生成带有价格波动的船只。
    这样才能体现出 AI "挑肥拣瘦" 的优化能力。
    """
    ships = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = rng.integers(0, main.TOTAL_STEPS - 20)
        t_c = rng.integers(8, 12)
        t_sp = t_c + rng.integers(4, 8)
        t_bs = 1
        ddl = min(main.TOTAL_STEPS, arr + t_sp + 10)

        # 🔥 核心修改：制造价格差异 (Heterogeneity)
        # 基础换电价格是 120，我们让它在 60 到 200 之间波动
        # 岸电价格固定 50
        real_batt_cost = rng.integers(60, 250)

        # 构造船只 (注意：这里把 real_batt_cost 传进去)
        ships.append(main.Ship(i, arr, ddl, t_c, t_sp, t_bs, 50.0, float(real_batt_cost), 1000.0))
    return ships


# 💉 注入补丁：强制 main.py 使用我们的新生成函数
main.generate_ships = generate_heterogeneous_ships
print(">>> ✅ 已注入异质性数据生成器 (VIP模式已开启)")


# ==========================================
# 0. 重新定义贪婪算法 (使用新的生成逻辑)
# ==========================================
def run_greedy_baseline(
    n_ships=20,
    n_sp=5,
    seed=42,
    logger=None,
    instance_id=None,
    scenario="heterogeneous",
    method_name="Greedy",
):
    # 直接调用被我们修改过的 generate_ships
    ships = main.generate_ships(n=n_ships, seed=seed)

    start_time = time.time()
    shore_schedule = np.zeros(TOTAL_STEPS)  # 资源占用表
    total_cost = 0

    # 贪婪策略：按到达时间排序 (First-Come-First-Served)
    sorted_ships = sorted(ships, key=lambda s: s.arrival_time)

    for s in sorted_ships:
        can_shore = False
        dur_shore = max(s.t_cargo, s.t_shore_power)
        end_time = s.arrival_time + dur_shore

        if end_time <= s.deadline and end_time <= TOTAL_STEPS:
            # 检查是否有空位
            conflict = False
            for t in range(s.arrival_time, end_time):
                if shore_schedule[t] >= n_sp:
                    conflict = True
                    break

            if not conflict:
                can_shore = True
                # 占用资源
                for t in range(s.arrival_time, end_time):
                    shore_schedule[t] += 1
                total_cost += s.cost_shore  # 岸电通常是 50

        if not can_shore:
            dur_batt = max(s.t_cargo, s.t_battery_swap)
            if s.arrival_time + dur_batt <= s.deadline:
                total_cost += s.cost_battery
            else:
                total_cost += s.cost_brown

    result_payload = {
        "obj": total_cost,
        "time": time.time() - start_time
    }
    if logger is not None:
        logger.log_row(
            {
                "instance_id": instance_id or f"N{n_ships}_seed{seed}_{scenario}",
                "N": n_ships,
                "seed": seed,
                "scenario": scenario,
                "method": method_name,
                "obj": result_payload["obj"],
                "runtime_total": result_payload["time"],
                "runtime_rmp": 0.0,
                "runtime_pricing": 0.0,
                "status": "ok",
                "gap": None,
                "num_iters": 1,
                "num_pricing_calls": 0,
                "num_fallback_calls": 0,
                "num_columns_added": 0,
                "min_reduced_cost_last": None,
                "pricing_time_share": 0.0,
            }
        )
    return result_payload


# ==========================================
# 1. 运行综合对比实验 (制造稀缺性 + 差异性)
# ==========================================
def exp_comprehensive_comparison(seed=42, logger=None):
    print("\n>>> [最终实验] 启动全维度对比 (Exact vs Greedy vs AI)...")
    results = []
    scales = [20, 50, 100]

    for n in scales:
        print(f"  正在测试规模 N={n} ...")

        # 🔥 关键点：制造资源稀缺 (100艘船抢 10个桩)
        # 迫使算法在 VIP 和普通船之间做取舍
        tight_n_sp = max(1, n // 10)

        # 1. Greedy
        res_greedy = run_greedy_baseline(
            n_ships=n,
            n_sp=tight_n_sp,
            seed=seed,
            logger=logger,
            instance_id=f"N{n}_seed{seed}_heterogeneous",
            scenario="heterogeneous",
            method_name="Greedy Heuristic",
        )
        results.append({
            "Scale": n, "Method": "Greedy Heuristic",
            "Time": res_greedy['time'], "Cost": res_greedy['obj']
        })

        # 2. AI-CG
        res_ai = run_ai_column_generation(
            n_ships=n,
            enable_ai=True,
            n_sp=tight_n_sp,
            seed=seed,
            logger=logger,
            instance_id=f"N{n}_seed{seed}_heterogeneous",
            scenario="heterogeneous",
            method_name="GNN-Accelerated CG",
        )
        results.append({
            "Scale": n, "Method": "GNN-Accelerated CG",
            "Time": res_ai['time'], "Cost": res_ai['obj']
        })

        # 3. Exact-CG
        if n <= 100:
            res_exact = run_ai_column_generation(
                n_ships=n,
                enable_ai=False,
                n_sp=tight_n_sp,
                seed=seed,
                logger=logger,
                instance_id=f"N{n}_seed{seed}_heterogeneous",
                scenario="heterogeneous",
                method_name="Exact CG",
            )
            results.append({
                "Scale": n, "Method": "Exact CG",
                "Time": res_exact['time'], "Cost": res_exact['obj']
            })

    df = pd.DataFrame(results)
    df.to_csv("exp_final_comparison.csv", index=False)
    print("✅ 综合对比数据已保存！")


# ==========================================
# 2. 画图 (修复版布局)
# ==========================================
def plot_pareto_comparison():
    try:
        df = pd.read_csv("exp_final_comparison.csv")
        subset = df[df['Scale'] == 100]

        plt.figure(figsize=(8, 6))
        # 论文常用配色
        colors = {'Greedy Heuristic': '#999999', 'Exact CG': '#2ca02c', 'GNN-Accelerated CG': '#d62728'}
        markers = {'Greedy Heuristic': 'x', 'Exact CG': 's', 'GNN-Accelerated CG': '*'}

        max_time = subset['Time'].max()
        min_cost = subset['Cost'].min()
        max_cost = subset['Cost'].max()

        for method in subset['Method'].unique():
            row = subset[subset['Method'] == method].iloc[0]
            plt.scatter(row['Time'], row['Cost'], s=300, c=colors[method], marker=markers[method], label=method)

            # 智能标签位置
            offset_y = (max_cost - min_cost) * 0.05
            va = 'top' if 'GNN' in method else 'bottom'
            y_pos = row['Cost'] - offset_y if 'GNN' in method else row['Cost'] + offset_y

            plt.text(row['Time'], y_pos,
                     f"{method}\n({row['Time']:.2f}s, ${int(row['Cost'])})",
                     ha='center', va=va, fontsize=10, fontweight='bold')

        plt.xlabel("Computation Time (s) [Lower is Better]", fontweight='bold')
        plt.ylabel("Total Cost ($) [Lower is Better]", fontweight='bold')
        plt.title("Algorithm Performance Trade-off (N=100, Heterogeneous)", fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)

        # 设置范围，防止箭头撑爆
        plt.xlim(-0.5, max_time * 1.3)
        plt.ylim(min_cost * 0.95, max_cost * 1.05)

        # 箭头指向理想区域
        plt.annotate('Ideal Region\n(Fast & Cheap)',
                     xy=(0, min_cost),
                     xytext=(max_time * 0.6, min_cost + (max_cost - min_cost) * 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=11, fontweight='bold', color='blue')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("Fig_Pareto_Frontier_Final.png", dpi=300)
        print("✅ 最终帕累托图已生成: Fig_Pareto_Frontier_Final.png")

    except Exception as e:
        print(f"绘图失败: {e}")


if __name__ == "__main__":
    logger = ExperimentLogger("results.csv")
    exp_comprehensive_comparison(logger=logger)
    plot_pareto_comparison()
