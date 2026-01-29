import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment_logger import ExperimentLogger
from solvers.cg_solver import solve_cg
from solvers.greedy_solver import solve_greedy
from solvers.milp_solver import solve_milp


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


TIME_HORIZON = 24
TOTAL_STEPS = (TIME_HORIZON * 60) // 15
K_TOTAL = 5


PRESETS = {
    "correctness": {
        "Ns": [20, 30],
        "methods": ["cg", "milp300"],
        "scenarios": ["U", "P"],
        "seeds": list(range(1, 21)),
    },
    "main": {
        "Ns": [20, 50, 100, 200, 500],
        "methods": ["greedy", "milp60", "milp300", "cg"],
        "scenarios": ["U", "P", "M"],
        "seeds": list(range(1, 31)),
    },
}


def parse_range_list(value):
    values = []
    for part in value.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":")
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return values


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_existing_keys(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return set()
    existing = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            existing.add((int(row["N"]), int(row["seed"]), row["scenario"], row["method"]))
    return existing


def generate_ships(n, seed, scenario, total_steps, battery_cost=120.0):
    rng = np.random.default_rng(seed)
    ships = []
    slack = 10
    if scenario.endswith("_loose"):
        slack = 20
    if scenario.endswith("_tight"):
        slack = 5

    def draw_arrival():
        if scenario.startswith("P"):
            center = total_steps // 2
            return int(np.clip(rng.normal(center, total_steps * 0.15), 0, total_steps - 20))
        if scenario.startswith("M"):
            if rng.random() < 0.5:
                center = total_steps // 2
                return int(np.clip(rng.normal(center, total_steps * 0.15), 0, total_steps - 20))
            return int(rng.integers(0, total_steps - 20))
        return int(rng.integers(0, total_steps - 20))

    for i in range(n):
        arr = draw_arrival()
        t_c = int(rng.integers(8, 12))
        t_sp = t_c + int(rng.integers(4, 8))
        t_bs = 1
        ddl = min(total_steps, arr + t_sp + slack)
        ships.append(
            Ship(
                i,
                arr,
                ddl,
                t_c,
                t_sp,
                t_bs,
                50.0,
                battery_cost,
                1000.0,
            )
        )
    return ships


def run_single(method, instance):
    if method == "greedy":
        return solve_greedy(instance)
    if method == "cg":
        return solve_cg(instance, {"enable_ai": False, "seed": instance["seed"], "method_name": "cg"})
    if method == "milp60":
        return solve_milp(instance, 60, instance["seed"])
    if method == "milp300":
        return solve_milp(instance, 300, instance["seed"])
    raise ValueError(f"Unknown method: {method}")


def log_error(logger, instance, method, error):
    logger.log_row(
        {
            "instance_id": instance["instance_id"],
            "N": instance["N"],
            "seed": instance["seed"],
            "scenario": instance["scenario"],
            "method": method,
            "obj": None,
            "runtime_total": None,
            "runtime_rmp": None,
            "runtime_pricing": None,
            "status": f"error: {error}",
            "gap": None,
            "num_iters": None,
            "num_pricing_calls": None,
            "num_fallback_calls": None,
            "num_columns_added": None,
            "min_reduced_cost_last": None,
            "rc_added_last": None,
            "pricing_time_share": None,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ns", default="20", help="Comma-separated list of N values.")
    parser.add_argument("--seeds", default="1:1", help="Range a:b or list a,b,c.")
    parser.add_argument("--scenarios", default="U", help="Comma-separated scenario list.")
    parser.add_argument("--methods", default="greedy,cg", help="Comma-separated method list.")
    parser.add_argument("--out", default="results.csv", help="Output CSV path.")
    parser.add_argument("--preset", choices=PRESETS.keys())
    args = parser.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        Ns = preset["Ns"]
        seeds = preset["seeds"]
        scenarios = preset["scenarios"]
        methods = preset["methods"]
    else:
        Ns = parse_range_list(args.Ns)
        seeds = parse_range_list(args.seeds)
        scenarios = parse_list(args.scenarios)
        methods = parse_list(args.methods)

    logger = ExperimentLogger(args.out)
    existing = load_existing_keys(args.out)

    for n in Ns:
        for seed in seeds:
            for scenario in scenarios:
                ships = generate_ships(n, seed, scenario, TOTAL_STEPS)
                instance = {
                    "instance_id": f"N{n}_seed{seed}_{scenario}",
                    "N": n,
                    "seed": seed,
                    "scenario": scenario,
                    "n_sp": max(1, n // 10),
                    "k_total": K_TOTAL,
                    "total_steps": TOTAL_STEPS,
                    "battery_cost": 120.0,
                    "ships": ships,
                }
                for method in methods:
                    key = (n, seed, scenario, method)
                    if key in existing:
                        continue
                    try:
                        result = run_single(method, instance)
                        logger.log_row(
                            {
                                "instance_id": instance["instance_id"],
                                "N": n,
                                "seed": seed,
                                "scenario": scenario,
                                "method": method,
                                "obj": result.get("obj"),
                                "runtime_total": result.get("runtime_total"),
                                "runtime_rmp": result.get("runtime_rmp"),
                                "runtime_pricing": result.get("runtime_pricing"),
                                "status": result.get("status"),
                                "gap": result.get("gap"),
                                "num_iters": result.get("num_iters"),
                                "num_pricing_calls": result.get("num_pricing_calls"),
                                "num_fallback_calls": result.get("num_fallback_calls"),
                                "num_columns_added": result.get("num_columns_added"),
                                "min_reduced_cost_last": result.get("min_reduced_cost_last"),
                                "rc_added_last": result.get("rc_added_last"),
                                "pricing_time_share": result.get("pricing_time_share"),
                            }
                        )
                    except Exception as exc:
                        log_error(logger, instance, method, exc)
                    existing.add(key)


if __name__ == "__main__":
    main()
