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
    ships=None,
    time_limit=None,
    logger=None,
    instance_id=None,
    scenario="default",
    method_name=None,
):
    # 1. 生成算例
    if ships is None:
        ships = generate_ships(n=n_ships, cost_battery_val=battery_cost, seed=seed)
    else:
        n_ships = len(ships)

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
    stats = {'ai_hits': 0, 'ai_misses': 0, 'fallback_calls': 0}
    obj_history = []
    total_rmp_time = 0.0
    total_pricing_time = 0.0
    pricing_calls = 0
    columns_added = 0
    rc_added_last = None
    min_rc_last_iter = None

    # --- 循环开始 ---
    for it in range(1, 51):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break
        rmp_start = time.time()
        mp.optimize()
        total_rmp_time += time.time() - rmp_start
        if mp.Status != GRB.OPTIMAL: break
        obj_history.append(mp.ObjVal)

        pi = {s.id: cons_fulfill[s.id].Pi for s in ships}
        mu = [cons_sp[t].Pi for t in range(TOTAL_STEPS)]
        rho = [cons_bs[t].Pi for t in range(TOTAL_STEPS)]

        new_cols = 0
        min_rc_this_iter = None
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

                pricing_calls += 1
                pricing_start = time.time()
                result = solve_pricing_problem(s, pi[s.id], mu, rho, TOTAL_STEPS, allowed_mode=pred_mode)
                total_pricing_time += time.time() - pricing_start
                if result:
                    stats['ai_hits'] += 1
                else:
                    stats['ai_misses'] += 1

            # 策略 B: 精确算法兜底
            if not result:
                stats['fallback_calls'] += 1
                pricing_calls += 1
                pricing_start = time.time()
                result = solve_pricing_problem(s, pi[s.id], mu, rho, TOTAL_STEPS, allowed_mode='all')
                total_pricing_time += time.time() - pricing_start

            # 添加列
            if result is not None:
                rc_val = result.get("rc")
                if rc_val is not None:
                    if min_rc_this_iter is None or rc_val < min_rc_this_iter:
                        min_rc_this_iter = rc_val
                if "mode" in result:
                    rc_added_last = result["rc"]
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
                    columns_added += 1

        min_rc_last_iter = min_rc_this_iter
        # 如果没有新列，说明收敛，跳出循环
        if new_cols == 0:
            break

    # --- 循环结束 (注意：下面的代码必须顶格写，不能缩进在 for 里面！) ---

    end_time = time.time()
    total_time = end_time - start_time
    pricing_time_share = (total_pricing_time / total_time) if total_time > 0 else 0.0

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

    result_payload = {
        "time": total_time,
        "obj": mp.ObjVal,
        "iter": len(obj_history),
        "shore_rate": shore_rate,
        "history": obj_history,
        "runtime_rmp": total_rmp_time,
        "runtime_pricing": total_pricing_time,
        "pricing_calls": pricing_calls,
        "fallback_calls": stats["fallback_calls"],
        "columns_added": columns_added,
        "min_reduced_cost_last": min_rc_last_iter,
        "rc_added_last": rc_added_last,
        "pricing_time_share": pricing_time_share,
        "status": mp.Status,
    }
    if logger is not None:
        method = method_name or ("AI-CG" if enable_ai else "Exact CG")
        status = mp.Status
        gap = None
        if hasattr(mp, "ObjBound") and mp.ObjVal != 0:
            gap = abs(mp.ObjVal - mp.ObjBound) / abs(mp.ObjVal)
        logger.log_row(
            {
                "instance_id": instance_id or f"N{n_ships}_seed{seed}_{scenario}",
                "N": n_ships,
                "seed": seed,
                "scenario": scenario,
                "method": method,
                "obj": mp.ObjVal,
                "runtime_total": total_time,
                "runtime_rmp": total_rmp_time,
                "runtime_pricing": total_pricing_time,
                "status": status,
                "gap": gap,
                "num_iters": len(obj_history),
                "num_pricing_calls": pricing_calls,
                "num_fallback_calls": stats["fallback_calls"],
                "num_columns_added": columns_added,
                "min_reduced_cost_last": min_rc_last_iter,
                "rc_added_last": rc_added_last,
                "pricing_time_share": pricing_time_share,
            }
        )
    return result_payload


if __name__ == "__main__":
    # 简单自测
    print(">>> 正在进行一次自测运行...")
    res = run_ai_column_generation(n_ships=20, enable_ai=True)
    print(f">>> 自测完成: 耗时 {res['time']:.4f}s, 目标值 {res['obj']:.2f}")
