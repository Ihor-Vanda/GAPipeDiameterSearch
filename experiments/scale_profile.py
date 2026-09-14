"""Куди йде бюджет симуляцій і як це масштабується з розміром мережі.

Питання, на яке відповідає цей замір: на великих мережах майже весь
бюджет з'їдають ініціалізація й променевий пошук, і до адаптивної
машинерії кіків прогін не доходить. Тут міряється, скільки саме коштує
кожна фаза і як цей коштунок росте з кількістю труб.

Міряється по фазах:

* `sim_rate`      — скільки симуляцій на секунду дає симулятор сам собою;
* `seed`          — побудова початкових рішень (насіння) + їх ремонт;
* `round`         — один раунд променевого пошуку (мутації + поліровка);
* `to_first_kick` — оцінка, скільки симуляцій треба до першого кіка.

Прогін навмисно обривається після заданої кількості раундів: щоб
зміряти ціну фази, не потрібно доводити пошук до кінця.

    python experiments/scale_profile.py --sizes 10 14 20 24 --rounds 2
    python experiments/scale_profile.py --networks InputData/Hanoi/Hanoi.inp:in:30
"""

import argparse
import csv
import os
import random
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from analytical_solver import orchestrator  # noqa: E402
from analytical_solver.context import (  # noqa: E402
    SolverContext, beam_width_for, classify_network,
)
from analytical_solver.kicks import KickStrategies  # noqa: E402
from analytical_solver.local_search import LocalSearch  # noqa: E402
from make_grid_network import build  # noqa: E402
from config import SolverConfig  # noqa: E402
from cost_table import load_cost_table  # noqa: E402
from water_sim import WaterSimulator  # noqa: E402

BALERMA_COSTS = os.path.join("InputData", "Balerma", "costs.csv")


class Stop(Exception):
    """Достатньо раундів — далі міряти нічого."""


class ProfiledWorker(orchestrator.IslandWorker):
    """Воркер, який записує ціну кожної фази в симуляціях і секундах."""

    def __init__(self, *args, wall_cap=180.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_cap = wall_cap
        self.started_at = time.perf_counter()
        self.phases = {"seed": [0, 0.0], "mutate": [0, 0.0], "beam": [0, 0.0],
                       "kick": [0, 0.0]}
        #: Скільки симуляцій витрачено до першого кіка — головне число.
        self.sims_at_first_kick = None
        self.rounds_done = 0
        self.kicks_done = 0
        self._wrap_seeder()

    def _check_cap(self):
        if time.perf_counter() - self.started_at > self.wall_cap:
            raise Stop

    def _wrap_seeder(self):
        """Насіння будується в `run()` напряму через `seeder`, не в методі воркера."""
        for name in ("make_diverse_seeds", "make_warm_seeds", "make_reserve_pool"):
            original = getattr(self.seeder, name, None)
            if original is None:
                continue

            def wrapped(*args, _original=original, **kwargs):
                return self._record("seed", _original, *args, **kwargs)

            setattr(self.seeder, name, wrapped)

    def _record(self, phase, method, *args, **kwargs):
        sims_before = self.ctx.sim_count
        started = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            self.phases[phase][0] += self.ctx.sim_count - sims_before
            self.phases[phase][1] += time.perf_counter() - started

    def _generate_mutations(self):
        return self._record("mutate", super()._generate_mutations)

    def _beam_search_and_update(self, *args, **kwargs):
        result = self._record("beam", super()._beam_search_and_update, *args, **kwargs)
        self.rounds_done += 1
        self._check_cap()
        return result

    def _apply_kick(self, *args, **kwargs):
        if self.sims_at_first_kick is None:
            self.sims_at_first_kick = self.ctx.sim_count
        self.kicks_done += 1
        result = self._record("kick", super()._apply_kick, *args, **kwargs)
        self._check_cap()
        return result


def measure_sim_rate(simulator, num_pipes, samples=150):
    """Чиста пропускна здатність симулятора, без солвера."""
    options = simulator.n_options
    vectors = [[random.randrange(options) for _ in range(num_pipes)]
               for _ in range(samples)]
    started = time.perf_counter()
    for vector in vectors:
        simulator.get_stats(vector)
    elapsed = time.perf_counter() - started
    return samples / elapsed if elapsed > 0 else float("inf")


def profile(inp_file, cost_file, h_min, units, max_sims, seed, sim_samples,
            wall_cap):
    random.seed(seed)
    np.random.seed(seed)

    config = SolverConfig(cost_file=os.path.abspath(cost_file), h_min=h_min,
                      unit_system=units, run_mode="analytical", v_opt=1.0)
    load_cost_table(config)

    temp_dir = os.path.abspath(os.path.join("_temp_sim_data", "scale"))
    os.makedirs(temp_dir, exist_ok=True)
    simulator = WaterSimulator(os.path.abspath(inp_file), config, temp_dir=temp_dir)
    num_pipes = simulator.n_variables

    sim_rate = measure_sim_rate(simulator, num_pipes, samples=sim_samples)

    ctx = SolverContext(simulator, config.diameters_m, v_opt=1.0)
    local_search = LocalSearch(ctx)
    network_class = classify_network(num_pipes)

    worker = ProfiledWorker(
        ctx, KickStrategies(ctx, local_search), local_search, 0, 1,
        max_sims, beam_width_for(network_class), network_class, [],
        wall_cap=wall_cap,
    )

    started = time.perf_counter()
    try:
        worker.run(float("inf"), float("inf"), None)
    except Stop:
        pass
    wall = time.perf_counter() - started

    total = max(1, ctx.sim_count)
    seed_sims, seed_time = worker.phases["seed"]
    beam_sims = worker.phases["mutate"][0] + worker.phases["beam"][0]
    kick_sims = worker.phases["kick"][0]

    return {
        "network": os.path.splitext(os.path.basename(inp_file))[0],
        "pipes": num_pipes,
        "class": network_class,
        "sim_rate": round(sim_rate, 1),
        "seed_sims": seed_sims,
        "seed_s": round(seed_time, 2),
        "seed_pct": round(100 * seed_sims / total, 1),
        "beam_sims": beam_sims,
        "beam_pct": round(100 * beam_sims / total, 1),
        "rounds": worker.rounds_done,
        "sims_per_round": int(beam_sims / max(1, worker.rounds_done)),
        "kick_sims": kick_sims,
        "kick_pct": round(100 * kick_sims / total, 1),
        "kicks": worker.kicks_done,
        "sims_to_first_kick": worker.sims_at_first_kick,
        "total_sims": ctx.sim_count,
        "wall_s": round(wall, 1),
    }


def grid_network(side, out_root="_temp_sim_data/nets"):
    """Згенерувати сітку side×side і повернути шлях до .inp."""
    text, n_pipes, _ = build(side, side)
    folder = os.path.join(out_root, f"Grid{side}x{side}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"Grid{side}x{side}.inp")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path, n_pipes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[10, 14, 20, 24],
                        help="Сторони синтетичних сіток (side × side вузлів)")
    parser.add_argument("--networks", nargs="*", default=[],
                        help="Реальні мережі у вигляді шлях:одиниці:h_min")
    parser.add_argument("--max_sims", type=int, default=20000)
    parser.add_argument("--wall_cap", type=float, default=180.0,
                        help="Стеля часу на одну мережу, с")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim_samples", type=int, default=150)
    parser.add_argument("--out", default="experiments/scale_profile.csv")
    args = parser.parse_args()

    jobs = []
    for spec in args.networks:
        path, units, h_min = spec.split(":")
        costs = os.path.join(os.path.dirname(path), "costs.csv")
        jobs.append((path, costs, float(h_min), units))
    for side in args.sizes:
        path, _ = grid_network(side)
        jobs.append((path, BALERMA_COSTS, 20.0, "mm"))

    rows = []
    for inp_file, cost_file, h_min, units in jobs:
        row = profile(inp_file, cost_file, h_min, units, args.max_sims,
                      args.seed, args.sim_samples, args.wall_cap)
        rows.append(row)
        first = row["sims_to_first_kick"]
        print(f"  {row['network']:<14} труб={row['pipes']:<5} {row['class']:<7} "
              f"{row['sim_rate']:>7.1f} сим/с | насіння {row['seed_pct']:>5.1f} % "
              f"({row['seed_sims']:>6} сим, {row['seed_s']:>5.1f} с) | променевий "
              f"{row['beam_pct']:>5.1f} % ({row['rounds']:>3} раунд., "
              f"{row['sims_per_round']:>6} сим/раунд) | кіки {row['kick_pct']:>5.1f} % "
              f"({row['kicks']:>4}) | до 1-го кіка "
              f"{first if first is not None else '—':>8} | {row['wall_s']:>6.1f} с",
              flush=True)

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  {args.out}")


if __name__ == "__main__":
    main()
