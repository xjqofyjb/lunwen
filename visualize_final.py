import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch
import joblib
import time
import networkx as nx

# 复用你现有的模块
from main import generate_ships, PricingPredictor, TOTAL_STEPS, N_SP, K_TOTAL
from pricing_algo import solve_pricing_problem

# 设置论文绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def run_single_visualization_case():
    print(">>> [可视化] 正在运行单次算例以生成甘特图和收敛曲线...")

    # 1. 准备环境 (复用 main.py 的逻辑，但为了提取数据稍微改写一点)
    ships = generate_ships(n=15, cost_battery_val=120.0)  # 15艘船画出来最清晰

    # 加载 AI
    model = PricingPredictor(input_dim=4, output_dim=2)
    model.load_state_dict(torch.load("pricing_model.pth"))
    model.eval()
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    mp = gp.Model("Vis_Master")
    mp.Params.OutputFlag = 0

    z_dummies = {}
    for s in ships:
        z_dummies[s.id] = mp.addVar(obj=s.cost_brown, vtype=GRB.CONTINUOUS, name=f"z_{s.id}")
    mp.update()

    cons_fulfill = {s.id: mp.addConstr(z_dummies[s.id] == 1) for s in ships}
    cons_sp = {t: mp.addConstr(gp.LinExpr() <= N_SP) for t in range(TOTAL_STEPS)}
    cons_bs = {t: mp.addConstr(gp.LinExpr() <= K_TOTAL) for t in range(TOTAL_STEPS)}

    # 记录收敛历史
    history = []

    # 列生成主循环
    for it in range(1, 30):
        mp.optimize()
        obj = mp.ObjVal
        history.append({'Iter': it, 'Obj': obj})

        pi = {s.id: cons_fulfill[s.id].Pi for s in ships}
        mu = [cons_sp[t].Pi for t in range(TOTAL_STEPS)]
        rho = [cons_bs[t].Pi for t in range(TOTAL_STEPS)]

        new_cols = 0
        for s in ships:
            # 简化：直接用 AI 预测 + 兜底
            mu_avg = np.mean(mu)
            features = np.array([[s.t_cargo, s.t_shore_power, pi[s.id], mu_avg]])
            feat_scaled = scaler.transform(features)
            with torch.no_grad():
                pred_idx = torch.argmax(model(torch.FloatTensor(feat_scaled)), dim=1).item()
                mode = label_encoder.inverse_transform([pred_idx])[0]

            res = solve_pricing_problem(s, pi[s.id], mu, rho, TOTAL_STEPS, allowed_mode=mode)
            if not res: res = solve_pricing_problem(s, pi[s.id], mu, rho, TOTAL_STEPS, allowed_mode='all')

            if res:
                col = gp.Column()
                col.addTerms(1.0, cons_fulfill[s.id])
                if res['mode'] == 'shore':
                    for k in range(res['start'], res['start'] + res['duration']):
                        if k < TOTAL_STEPS: col.addTerms(1.0, cons_sp[k])
                elif res['mode'] == 'battery':
                    if res['start'] < TOTAL_STEPS: col.addTerms(1.0, cons_bs[res['start']])

                # 关键：把 mode 和 start 存在变量名里，方便后面解析
                vname = f"x_{s.id}_{res['mode']}_{res['start']}_{res['duration']}"
                mp.addVar(obj=res['cost'], vtype=GRB.CONTINUOUS, column=col, name=vname)
                new_cols += 1

        if new_cols == 0: break

    # 求解整数规划以获取最终调度方案
    for v in mp.getVars(): v.vtype = GRB.BINARY
    mp.optimize()

    # 解析调度方案
    schedule = []
    for v in mp.getVars():
        if v.x > 0.5 and "x_" in v.VarName:
            # 变量名格式: x_id_mode_start_duration
            parts = v.VarName.split('_')
            sid, mode, start, dur = int(parts[1]), parts[2], int(parts[3]), int(parts[4])
            schedule.append({
                'Ship': sid,
                'Mode': mode,
                'Start': start,
                'Duration': dur,
                'End': start + dur
            })

    return pd.DataFrame(history), pd.DataFrame(schedule)


def plot_charts(df_hist, df_sched):
    # --- 图 1: 收敛曲线 (Convergence) ---
    plt.figure(figsize=(8, 5))
    plt.plot(df_hist['Iter'], df_hist['Obj'], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.xlabel('CG Iteration', fontweight='bold')
    plt.ylabel('Objective Value (Total Cost)', fontweight='bold')
    plt.title('Convergence Profile of Column Generation', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('Fig_Convergence.png', dpi=300, bbox_inches='tight')
    print("🖼️ 收敛曲线已生成: Fig_Convergence.png")

    # --- 图 2: 调度甘特图 (Gantt Chart) ---
    plt.figure(figsize=(10, 6))

    # 颜色映射
    colors = {'shore': '#2ca02c', 'battery': '#d62728'}  # 绿=岸电, 红=换电
    labels = {'shore': 'Shore Power', 'battery': 'Battery Swap'}

    # 为了图例不重复，记录是否画过
    seen_labels = set()

    # 按船ID排序
    df_sched = df_sched.sort_values('Ship', ascending=False)

    for idx, row in df_sched.iterrows():
        lbl = labels[row['Mode']] if row['Mode'] not in seen_labels else ""
        seen_labels.add(row['Mode'])

        plt.barh(
            y=f"Ship {row['Ship']}",
            width=row['Duration'],
            left=row['Start'],
            color=colors[row['Mode']],
            edgecolor='black',
            alpha=0.8,
            label=lbl,
            height=0.6
        )

        # 标一下到达时间 (Arr)
        # 这里简单处理，实际上最好把 arrival 也传出来，这里只画调度条

    plt.xlabel('Time Steps (15 min/step)', fontweight='bold')
    plt.ylabel('Vessel ID', fontweight='bold')
    plt.title('Optimal Energy Scheduling Gantt Chart', fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.savefig('Fig_Gantt_Schedule.png', dpi=300, bbox_inches='tight')
    print("🖼️ 甘特图已生成: Fig_Gantt_Schedule.png")


if __name__ == "__main__":
    hist, sched = run_single_visualization_case()
    plot_charts(hist, sched)