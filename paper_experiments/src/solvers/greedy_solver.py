import time


def solve_greedy(instance):
    ships = instance["ships"]
    n_sp = instance.get("n_sp", 5)
    total_steps = instance.get("total_steps")
    if total_steps is None:
        raise ValueError("instance must include total_steps")

    start_time = time.time()
    shore_schedule = [0] * total_steps
    total_cost = 0.0

    sorted_ships = sorted(ships, key=lambda s: s.arrival_time)
    for s in sorted_ships:
        can_shore = False
        dur_shore = max(s.t_cargo, s.t_shore_power)
        end_time = s.arrival_time + dur_shore

        if end_time <= s.deadline and end_time <= total_steps:
            conflict = False
            for t in range(s.arrival_time, end_time):
                if shore_schedule[t] >= n_sp:
                    conflict = True
                    break
            if not conflict:
                can_shore = True
                for t in range(s.arrival_time, end_time):
                    shore_schedule[t] += 1
                total_cost += s.cost_shore

        if not can_shore:
            dur_batt = max(s.t_cargo, s.t_battery_swap)
            if s.arrival_time + dur_batt <= s.deadline:
                total_cost += s.cost_battery
            else:
                total_cost += s.cost_brown

    return {
        "obj": total_cost,
        "runtime_total": time.time() - start_time,
        "status": "ok",
    }
