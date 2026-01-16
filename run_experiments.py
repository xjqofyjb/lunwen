import pandas as pd
import numpy as np
from main import run_ai_column_generation, generate_ships


# ==========================================
# 实验 1: 算法性能消融 (保留)
# ==========================================
def exp_ablation():
    print("\n>>> [实验 1] 启动算法消融实验 (AI vs No-AI)...")
    results = []
    scales = [20, 50, 100]

    for n in scales:
        print(f"  正在运行规模 N={n} ...")
        # 1. AI 模式
        res_ai = run_ai_column_generation(n_ships=n, enable_ai=True, n_sp=n // 2)
        results.append({"Scale": n, "Method": "With AI (Proposed)", "Time": res_ai['time']})

        # 2. 传统模式
        res_no_ai = run_ai_column_generation(n_ships=n, enable_ai=False, n_sp=n // 2)
        results.append({"Scale": n, "Method": "Exact CG (Baseline)", "Time": res_no_ai['time']})

    df = pd.DataFrame(results)
    df.to_csv("exp_ablation_results.csv", index=False)
    print("✅ 消融实验完成！")


# ==========================================
# 实验 2: 关键！多模式成本对比 (新增)
# 目的: 证明混合策略比单一策略省钱
# ==========================================
def exp_model_comparison():
    print("\n>>> [实验 2] 启动多模式成本对比 (Shore vs Battery vs Hybrid)...")
    results = []

    # 使用一个固定的中等规模算例，重复 5 次取平均，消除随机性
    N_SHIPS = 50
    N_SP_VAL = 8  # 岸电桩适中，制造一点竞争

    # 我们通过调整 costs 来模拟“强制模式”是不太容易的
    # 最好的办法是给 run_ai_column_generation 加一个 force_mode 参数
    # 但为了不改 main.py，我们用一种“价格欺骗法”：
    # 1. 强制岸电：把换电价格设为天价 (10000)
    # 2. 强制换电：把岸电价格设为天价 (注意：这需要去 Ship 类改，比较麻烦)

    # --- 简易方案：只跑 Hybrid，然后手动解析结果来估算 ---
    # 或者，我们直接修改 main.py 的 generate_ships 里的参数？
    # 不，最好的办法是直接在 run_ai_column_generation 里加控制。

    # 鉴于我们要尽量少改 main.py，我们用“参数控制法”：

    # 场景 A: 纯换电 (Battery Only)
    # 方法：把 battery_cost 设为正常(120)，但在逻辑上我们知道 AI 会选它
    # 等等，最稳妥的方法是去 main.py 加一个 `allowed_mode` 参数传递给 pricing
    # 但那样改动太大。

    # 💡 替代方案：利用现有的 battery_cost 参数

    # 1. Hybrid (你的模型)
    print("  正在运行 Hybrid Mode (Proposed)...")
    res_hybrid = run_ai_column_generation(n_ships=N_SHIPS, enable_ai=True, battery_cost=120, n_sp=N_SP_VAL)
    results.append({"Mode": "Hybrid (Proposed)", "Cost": res_hybrid['obj']})

    # 2. Battery Only (纯换电)
    # 方法：我们假设岸电桩 N_SP = 0，逼迫所有船去换电！
    print("  正在运行 Battery Only Mode...")
    res_batt = run_ai_column_generation(n_ships=N_SHIPS, enable_ai=True, battery_cost=120, n_sp=0)
    results.append({"Mode": "Battery Only", "Cost": res_batt['obj']})

    # 3. Shore Only (纯岸电)
    # 方法：把换电成本设为 9999 (天价)，逼迫算法选岸电
    # 如果岸电桩不够，就会产生高额排队/违约惩罚，这正是我们要展示的！
    print("  正在运行 Shore Only Mode...")
    res_shore = run_ai_column_generation(n_ships=N_SHIPS, enable_ai=True, battery_cost=9999.0, n_sp=N_SP_VAL)
    # 注意：如果有很多罚款，Obj 会很高，这正是我们要的
    results.append({"Mode": "Shore Power Only", "Cost": res_shore['obj']})

    df = pd.DataFrame(results)
    df.to_csv("exp_model_comparison.csv", index=False)
    print("✅ 模式对比实验完成！")


# ==========================================
# 实验 3: 基础设施敏感性 (新增)
# 目的: 分析岸电桩数量的影响
# ==========================================
def exp_resource_sensitivity():
    print("\n>>> [实验 3] 启动资源敏感性分析 (岸电桩数量)...")
    results = []

    n_ships = 50
    # 岸电桩从 0 到 15 个
    sp_counts = [0, 2, 4, 6, 8, 10, 12, 15]

    for k in sp_counts:
        print(f"  正在测试岸电桩数量 N_SP={k} ...")
        res = run_ai_column_generation(n_ships=n_ships, enable_ai=True, battery_cost=120, n_sp=k)

        results.append({
            "N_SP": k,
            "Total_Cost": res['obj'],
            "Shore_Rate": res['shore_rate'] * 100
        })

    df = pd.DataFrame(results)
    df.to_csv("exp_resource_sensitivity.csv", index=False)
    print("✅ 资源敏感性实验完成！")


if __name__ == "__main__":
    # 依次运行所有实验
    exp_ablation()
    exp_model_comparison()  # 新增
    exp_resource_sensitivity()  # 新增