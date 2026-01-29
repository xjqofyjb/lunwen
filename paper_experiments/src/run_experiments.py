import argparse
import csv
from pathlib import Path

from experiment_logger import ExperimentLogger, SCHEMA
from instances import generate_instance
from solvers.cg_solver import solve_cg
from solvers.greedy_solver import solve_fifo, solve_greedy
from solvers.milp_solver import solve_milp


def parse_config(path):
    config = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            config[key] = parse_value(value)
    return config


def parse_value(value):
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [item.strip() for item in items if item.strip()]
    if ".." in value:
        start, end = value.split("..")
        return list(range(int(start), int(end) + 1))
    if value.isdigit():
        return int(value)
    return value


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


def run_method(method, instance):
    if method == "greedy":
        return solve_greedy(instance)
    if method == "fifo":
        return solve_fifo(instance)
    if method == "cg":
        return solve_cg(instance, {"enable_ai": False, "seed": instance["seed"], "method_name": "cg"})
    if method == "milp60":
        return solve_milp(instance, 60, instance["seed"])
    if method == "milp300":
        return solve_milp(instance, 300, instance["seed"])
    raise ValueError(f"Unknown method: {method}")


def log_error(logger, instance, method, error):
    row = {key: None for key in SCHEMA}
    row.update(
        {
            "id": instance["id"],
            "N": instance["N"],
            "seed": instance["seed"],
            "scenario": instance["scenario"],
            "method": method,
            "status": f"error: {error}",
        }
    )
    logger.log_row(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = parse_config(args.config)
    Ns = [int(n) for n in config["Ns"]]
    seeds = [int(s) for s in config["seeds"]]
    scenarios = config["scenarios"]
    methods = config["methods"]
    out_name = config["out"]

    out_path = Path("results") / out_name
    logger = ExperimentLogger(out_path)
    existing = load_existing_keys(out_path)

    for n in Ns:
        for seed in seeds:
            for scenario in scenarios:
                try:
                    instance = generate_instance(n, seed, scenario)
                except Exception as exc:
                    for method in methods:
                        log_error(logger, {"id": f"N{n}_seed{seed}_{scenario}", "N": n, "seed": seed, "scenario": scenario}, method, exc)
                    continue
                for method in methods:
                    key = (n, seed, scenario, method)
                    if key in existing:
                        continue
                    try:
                        result = run_method(method, instance)
                        row = {key: None for key in SCHEMA}
                        row.update(
                            {
                                "id": instance["id"],
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
                                "pricing_time_share": result.get("pricing_time_share"),
                            }
                        )
                        logger.log_row(row)
                    except Exception as exc:
                        log_error(logger, instance, method, exc)
                    existing.add(key)


if __name__ == "__main__":
    main()
