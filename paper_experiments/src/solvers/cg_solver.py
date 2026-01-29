import sys
from pathlib import Path


def _import_cg():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from main import run_ai_column_generation
    except Exception as exc:
        raise RuntimeError(
            "Failed to import CG solver. Ensure repository root is on PYTHONPATH."
        ) from exc
    return run_ai_column_generation


def solve_cg(instance, params):
    run_ai_column_generation = _import_cg()
    seed = params.get("seed", instance.get("seed", 42))
    result = run_ai_column_generation(
        n_ships=instance.get("N"),
        enable_ai=params.get("enable_ai", False),
        battery_cost=instance.get("battery_cost", 120.0),
        n_sp=instance.get("n_sp", 5),
        seed=seed,
        ships=instance.get("ships"),
        time_limit=params.get("time_limit"),
        logger=None,
        instance_id=instance.get("id"),
        scenario=instance.get("scenario", "U"),
        method_name=params.get("method_name", "cg"),
    )
    runtime_total = result.get("time", 0.0)
    runtime_pricing = result.get("runtime_pricing", 0.0)
    pricing_time_share = runtime_pricing / runtime_total if runtime_total else 0.0
    return {
        "obj": result.get("obj"),
        "runtime_total": runtime_total,
        "runtime_rmp": result.get("runtime_rmp"),
        "runtime_pricing": runtime_pricing,
        "status": result.get("status"),
        "num_iters": result.get("iter"),
        "num_pricing_calls": result.get("pricing_calls"),
        "num_fallback_calls": 0,
        "num_columns_added": result.get("columns_added"),
        "min_reduced_cost_last": result.get("min_reduced_cost_last"),
        "pricing_time_share": pricing_time_share,
    }
