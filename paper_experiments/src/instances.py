from dataclasses import dataclass
from typing import List


TIME_HORIZON = 24
TOTAL_STEPS = (TIME_HORIZON * 60) // 15
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


def _generate_ships_fallback(n: int, seed: int, cost_battery_val: float) -> List[Ship]:
    import random

    rng = random.Random(seed)
    ships = []
    for i in range(n):
        arr = rng.randrange(0, TOTAL_STEPS - 20)
        t_c = rng.randrange(8, 12)
        t_sp = t_c + rng.randrange(4, 8)
        t_bs = 1
        ddl = min(TOTAL_STEPS, arr + t_sp + 10)
        ships.append(
            Ship(i, arr, ddl, t_c, t_sp, t_bs, 50.0, cost_battery_val, 1000.0)
        )
    return ships


def generate_instance(N: int, seed: int, scenario: str):
    if scenario != "U":
        raise NotImplementedError(f"Scenario '{scenario}' not implemented")

    try:
        from main import generate_ships as legacy_generate_ships
    except Exception:
        legacy_generate_ships = None

    if legacy_generate_ships is not None:
        ships = legacy_generate_ships(n=N, cost_battery_val=120.0, seed=seed)
    else:
        ships = _generate_ships_fallback(N, seed, 120.0)

    return {
        "id": f"N{N}_seed{seed}_{scenario}",
        "N": N,
        "seed": seed,
        "scenario": scenario,
        "n_sp": max(1, N // 10),
        "k_total": K_TOTAL,
        "total_steps": TOTAL_STEPS,
        "battery_cost": 120.0,
        "ships": ships,
    }
