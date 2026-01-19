import time

import gurobipy as gp
from gurobipy import GRB


def solve_milp(instance, timelimit_seconds, seed):
    ships = instance["ships"]
    n_sp = instance.get("n_sp", 5)
    k_total = instance.get("k_total", 5)
    total_steps = instance.get("total_steps")
    if total_steps is None:
        raise ValueError("instance must include total_steps")

    model = gp.Model("MILP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timelimit_seconds
    model.Params.Seed = seed
    model.Params.Threads = 1

    x_shore = {}
    x_batt = {}
    for s in ships:
        for t in range(s.arrival_time, s.deadline):
            dur_shore = max(s.t_cargo, s.t_shore_power)
            dur_batt = max(s.t_cargo, s.t_battery_swap)
            if t + dur_shore <= s.deadline and t + dur_shore <= total_steps:
                x_shore[(s.id, t)] = model.addVar(vtype=GRB.BINARY, obj=s.cost_shore)
            if t + dur_batt <= s.deadline and t + dur_batt <= total_steps:
                x_batt[(s.id, t)] = model.addVar(vtype=GRB.BINARY, obj=s.cost_battery)
    x_brown = {s.id: model.addVar(vtype=GRB.BINARY, obj=s.cost_brown) for s in ships}
    model.update()

    for s in ships:
        shore_vars = [x_shore[(s.id, t)] for t in range(s.arrival_time, s.deadline) if (s.id, t) in x_shore]
        batt_vars = [x_batt[(s.id, t)] for t in range(s.arrival_time, s.deadline) if (s.id, t) in x_batt]
        model.addConstr(gp.quicksum(shore_vars) + gp.quicksum(batt_vars) + x_brown[s.id] == 1)

    for t in range(total_steps):
        shore_use = []
        batt_use = []
        for s in ships:
            dur_shore = max(s.t_cargo, s.t_shore_power)
            dur_batt = max(s.t_cargo, s.t_battery_swap)
            for start_t in range(s.arrival_time, s.deadline):
                if (s.id, start_t) in x_shore and start_t <= t < start_t + dur_shore:
                    shore_use.append(x_shore[(s.id, start_t)])
                if (s.id, start_t) in x_batt and start_t <= t < start_t + dur_batt:
                    batt_use.append(x_batt[(s.id, start_t)])
        if shore_use:
            model.addConstr(gp.quicksum(shore_use) <= n_sp)
        if batt_use:
            model.addConstr(gp.quicksum(batt_use) <= k_total)

    start_time = time.time()
    model.optimize()
    runtime_total = time.time() - start_time

    obj = None
    if model.SolCount > 0:
        obj = model.ObjVal

    return {
        "obj": obj,
        "runtime_total": runtime_total,
        "status": model.Status,
        "gap": model.MIPGap if model.SolCount > 0 else None,
    }
