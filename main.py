import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import pandas as pd
from dataclasses import dataclass
import torch
import torch.nn as nn
import joblib
import time

# 导入修改后的子问题
from pricing_algo import solve_pricing_problem


# ==========================================
# 0. 神经网络类定义
# ==========================================
class PricingPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PricingPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 1. 系统初始化与数据结构
# ==========================================
print(">>> [系统初始化] 正在加载 AI 模型...")
try:
    model = PricingPredictor(input_dim=4, output_dim=2)
    model.load_state_dict(torch.load("pricing_model.pth"))
    model.eval()
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    AI_ENABLED = True
    print("✅ AI 模型加载成功！将启用加速策略。")
except Exception as e:
    print(f"⚠️ 模型加载失败 ({e})，将回退到纯精确算法。")
    AI_ENABLED = False

TIME_HORIZON = 24
TOTAL_STEPS = (TIME_HORIZON * 60) // 15
N_SP = 5
K_TOTAL = 5


@dataclass
class Ship:
    id: int
    arrival_time: int
    deadline: int
    t_cargo: int
    t_shore_power: int
    t_battery_swap: int
    cost_shore: float
    cost_battery: float
    cost_brown: float


# ==========================================
# 2. 核心逻辑 (已修复缩进和统计 Bug)
# ==========================================
def generate_ships(n=20, cost_battery_val=120.0, seed=42):
    ships = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = rng.integers(0, TOTAL_STEPS - 20)
        t_c = rng.integers(8, 12)
        t_sp = t_c + rng.integers(4, 8)
        t_bs = 1
        ddl = min(TOTAL_STEPS, arr + t_sp + 10)
        # 注意：这里把 cost_battery 设为了传入的参数
        ships.append(Ship(i, arr, ddl, t_c, t_sp, t_bs, 50.0, cost_battery_val, 1000.0))
    return ships


def run_ai_column_generation(
    n_ships=20,
    enable_ai=True,
    battery_cost=120.0,
    n_sp=5,
    seed=42,
    time_limit=None,
):
    # 1. 生成算例
    ships = generate_ships(n=n_ships, cost_battery_val=battery_cost, seed=seed)

    mp = gp.Model("Master_AI")
    mp.Params.OutputFlag = 0  # 静默模式
    mp.Params.Seed = seed
    if time_limit is not None:
        mp.Params.TimeLimit = time_limit

    # 2. 初始化主问题
    z_dummies = {}
    for s in ships:
        z_dummies[s.id] = mp.addVar(obj=s.cost_brown, vtype=GRB.CONTINUOUS, name=f"z_{s.id}")
    mp.update()

    cons_fulfill = {s.id: mp.addConstr(z_dummies[s.id] == 1) for s in ships}
    cons_sp = {t: mp.addConstr(gp.LinExpr() <= n_sp) for t in range(TOTAL_STEPS)}
    cons_bs = {t: mp.addConstr(gp.LinExpr() <= K_TOTAL) for t in range(TOTAL_STEPS)}

    # 3. 列生成主循环
    start_time = time.time()
    stats = {'ai_hits': 0, 'ai_misses': 0}
    obj_history = []

    # --- 循环开始 ---
    for it in range(1, 51):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break
        mp.optimize()
        if mp.Status != GRB.OPTIMAL: break
        obj_history.append(mp.ObjVal)

        pi = {s.id: cons_fulfill[s.id].Pi for s in ships}
        mu = [cons_sp[t].Pi for t in range(TOTAL_STEPS)]
        rho = [cons_bs[t].Pi for t in range(TOTAL_STEPS)]

        new_cols = 0
        for s in ships:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break
            result = None

            # 策略 A: AI 预测
            if enable_ai and AI_ENABLED:
                mu_avg = np.mean(mu)
                features = np.array([[s.t_cargo, s.t_shore_power, pi[s.id], mu_avg]])
                feat_scaled = scaler.transform(features)
                feat_tensor = torch.FloatTensor(feat_scaled)
                with torch.no_grad():
                    logits = model(feat_tensor)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    pred_mode = label_encoder.inverse_transform([pred_idx])[0]

                result = solve_pricing_problem(s, pi[s.id], mu, rho, TOTAL_STEPS, allowed_mode=pred_mode)
                if result:
                    stats['ai_hits'] += 1
                else:
                    stats['ai_misses'] += 1

            # 策略 B: 精确算法兜底
            if not result:
                result = solve_pricing_problem(s, pi[s.id], mu, rho, TOTAL_STEPS, allowed_mode='all')

            # 添加列
            if result:
                col = gp.Column()
                col.addTerms(1.0, cons_fulfill[s.id])
                if result['mode'] == 'shore':
                    for k in range(result['start'], result['start'] + result['duration']):
                        if k < TOTAL_STEPS: col.addTerms(1.0, cons_sp[k])
                elif result['mode'] == 'battery':
                    if result['start'] < TOTAL_STEPS: col.addTerms(1.0, cons_bs[result['start']])

                # 变量名必须包含 shore/battery 方便后续统计
                var_name = f"x_{s.id}_{result['mode']}_{result['start']}"
                mp.addVar(obj=result['cost'], vtype=GRB.CONTINUOUS, column=col, name=var_name)
                new_cols += 1

        # 如果没有新列，说明收敛，跳出循环
        if new_cols == 0:
            break

    # --- 循环结束 (注意：下面的代码必须顶格写，不能缩进在 for 里面！) ---

    end_time = time.time()
    total_time = end_time - start_time

    # 4. 结果统计 (修复版)
    shore_count = 0
    if mp.Status == GRB.OPTIMAL:
        # 转换为整数规划以统计真实决策
        for v in mp.getVars():
            v.vtype = GRB.BINARY
        mp.optimize()

        for v in mp.getVars():
            # 只要变量名里包含 "shore" 且被选中，就计数
            if v.x > 0.5 and "shore" in v.VarName.lower():
                shore_count += 1

    shore_rate = shore_count / n_ships

    return {
        "time": total_time,
        "obj": mp.ObjVal,
        "iter": len(obj_history),
        "shore_rate": shore_rate,
        "history": obj_history
    }


if __name__ == "__main__":
    # 简单自测
    print(">>> 正在进行一次自测运行...")
    res = run_ai_column_generation(n_ships=20, enable_ai=True)
    print(f">>> 自测完成: 耗时 {res['time']:.4f}s, 目标值 {res['obj']:.2f}")
