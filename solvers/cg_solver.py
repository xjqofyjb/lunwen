def solve_cg(instance, params):
    from main import run_ai_column_generation

    enable_ai = params.get("enable_ai", False)
    seed = params.get("seed", instance.get("seed", 42))
    time_limit = params.get("time_limit")
    result = run_ai_column_generation(
        n_ships=instance.get("N"),
        enable_ai=enable_ai,
        battery_cost=instance.get("battery_cost", 120.0),
        n_sp=instance.get("n_sp", 5),
        seed=seed,
        ships=instance.get("ships"),
        time_limit=time_limit,
        logger=None,
        instance_id=instance.get("instance_id"),
        scenario=instance.get("scenario", "default"),
        method_name=params.get("method_name"),
    )
    return {
        "obj": result.get("obj"),
        "runtime_total": result.get("time"),
        "runtime_rmp": result.get("runtime_rmp"),
        "runtime_pricing": result.get("runtime_pricing"),
        "status": result.get("status"),
        "num_iters": result.get("iter"),
        "num_pricing_calls": result.get("pricing_calls"),
        "num_fallback_calls": result.get("fallback_calls"),
        "num_columns_added": result.get("columns_added"),
        "min_reduced_cost_last": result.get("min_reduced_cost_last"),
        "rc_added_last": result.get("rc_added_last"),
        "pricing_time_share": result.get("pricing_time_share"),
    }
