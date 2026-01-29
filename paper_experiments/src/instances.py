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


def _clustered_arrivals(n: int, seed: int):
    import random

    rng = random.Random(seed)
    peaks = [TOTAL_STEPS * 0.25, TOTAL_STEPS * 0.75]
    spread = TOTAL_STEPS * 0.08
    arrivals = []
    for _ in range(n):
        peak = peaks[0] if rng.random() < 0.5 else peaks[1]
        offset = rng.gauss(0, spread)
        arr = int(max(0, min(TOTAL_STEPS - 20, peak + offset)))
        arrivals.append(arr)
    return arrivals


def _apply_h1(ships, seed):
    arrivals = _clustered_arrivals(len(ships), seed)
    for ship, arr in zip(ships, arrivals):
        ship.arrival_time = arr
        duration = max(ship.t_cargo, ship.t_shore_power)
        ship.deadline = min(TOTAL_STEPS, arr + duration + 10)


def _apply_h2(ships, seed):
    import random

    rng = random.Random(seed)
    indices = list(range(len(ships)))
    rng.shuffle(indices)
    split = len(indices) // 2
    small_ids = set(indices[:split])
    for idx, ship in enumerate(ships):
        if idx in small_ids:
            ship.t_cargo = max(4, ship.t_cargo - 2)
            ship.t_shore_power = max(ship.t_cargo, ship.t_shore_power - 2)
            ship.cost_shore = max(30.0, ship.cost_shore - 10.0)
        else:
            ship.t_cargo = ship.t_cargo + 3
            ship.t_shore_power = ship.t_shore_power + 3
            ship.cost_shore = ship.cost_shore + 20.0
        duration = max(ship.t_cargo, ship.t_shore_power)
        ship.deadline = min(TOTAL_STEPS, ship.arrival_time + duration + 10)


def generate_instance(N: int, seed: int, scenario: str):
    if scenario not in {"U", "H1", "H2"}:
        raise NotImplementedError(f"Scenario '{scenario}' not implemented")

    try:
        from main import generate_ships as legacy_generate_ships
    except Exception:
        legacy_generate_ships = None

    if legacy_generate_ships is not None:
        ships = legacy_generate_ships(n=N, cost_battery_val=120.0, seed=seed)
    else:
        ships = _generate_ships_fallback(N, seed, 120.0)

    if scenario == "H1":
        _apply_h1(ships, seed)
    elif scenario == "H2":
        _apply_h2(ships, seed)

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
