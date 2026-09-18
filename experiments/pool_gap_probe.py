"""Замір складу пулу: чи справді він забивається дорогими рішеннями.

Питання, яке лишилось після етапу 7 (§20.9): фільтр розриву для великих
мереж був недосяжний, і перед тим як його повертати або викидати, треба
зміряти те, для чого він призначався.

Що міряється на кожному застосуванні кіка:

* відносний розрив **кожного** члена пулу до поточного рекорду;
* розрив члена, який справді обрано цілю кіка;
* розмір пулу і температура.

Ключова деталь, через яку сам фільтр був поставлений не в тому місці:
розрив члена пулу **не фіксований**. Він вимірюється до рекорду, а
рекорд падає, тому рішення, прийняте з розривом 5 %, згодом стає
відносно дорожчим, хоча його ніхто не перевіряє повторно. Фільтр діяв
лише на вході в пул, тому цього не ловив у принципі.

    python experiments/pool_gap_probe.py --inp InputData/Balerma/Balerma.inp \
        --costs InputData/Balerma/costs.csv --hmin 20 --max_sims 120000
"""

import argparse
import csv
import os
import random
import statistics
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from pipeopt.solver import island  # noqa: E402
from pipeopt.solver.context import (  # noqa: E402
    SolverContext, beam_width_for, classify_network,
)
from pipeopt.solver.kick_registry import KICK_REGISTRY  # noqa: E402
from pipeopt.solver.kicks import KickStrategies  # noqa: E402
from pipeopt.solver.local_search import LocalSearch  # noqa: E402
from pipeopt.config import SolverConfig  # noqa: E402
from pipeopt.reporting.cost_table import load_cost_table  # noqa: E402
from pipeopt.hydraulics.water_sim import WaterSimulator  # noqa: E402

#: Пороги, які перевіряються: задумані межі фільтра плюс ширші, щоб
#: побачити, чи розрив узагалі доростає до чогось помітного.
GAP_THRESHOLDS = (0.05, 0.08, 0.10, 0.15, 0.20)


class ProbedWorker(island.IslandWorker):
    """Воркер, який записує склад пулу перед кожним кіком."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = []
        self._target = None
        self._wrap_kicker()

    def _wrap_kicker(self):
        """Перехопити ціль кіка: це перший аргумент будь-якого методу кіка."""
        for spec in KICK_REGISTRY.values():
            original = getattr(self.kicker, spec.method, None)
            if original is None or getattr(original, "_probed", False):
                continue

            def probed(target, *args, _original=original, **kwargs):
                self._target = target
                return _original(target, *args, **kwargs)

            probed._probed = True
            setattr(self.kicker, spec.method, probed)

    def _apply_kick(self, round_idx, shared_progress):
        best_before = self.run_best_cost
        pool_before = [cost for _, cost, _ in self.pool.active_pool]
        temperature = min(1.0, self.stagnation_counter / max(1, self.stag_limit * 3.0))

        self._target = None
        super()._apply_kick(round_idx, shared_progress)

        if not pool_before or best_before <= 0 or best_before == float("inf"):
            return

        gaps = [(cost - best_before) / best_before for cost in pool_before]
        target_gap = None
        if self._target is not None:
            target_cost = self.ctx.get_cached_stats(list(self._target)).cost
            target_gap = (target_cost - best_before) / best_before

        self.samples.append({
            "round": round_idx,
            "temperature": round(temperature, 4),
            "pool_size": len(pool_before),
            "best_cost": best_before,
            "gap_max": max(gaps),
            "gap_mean": sum(gaps) / len(gaps),
            "gap_target": target_gap,
            "improved": int(self.run_best_cost < best_before),
            "sims": self.ctx.sim_count,
        })


def run_probe(inp_file, cost_file, h_min, max_sims, seed, units):
    random.seed(seed)
    np.random.seed(seed)

    config = SolverConfig(cost_file=os.path.abspath(cost_file), h_min=h_min,
                      unit_system=units, run_mode="analytical", v_opt=1.0)
    load_cost_table(config)

    temp_dir = os.path.abspath(os.path.join("_temp_sim_data", "probe"))
    os.makedirs(temp_dir, exist_ok=True)
    simulator = WaterSimulator(os.path.abspath(inp_file), config, temp_dir=temp_dir)

    ctx = SolverContext(simulator, config.diameters_m, v_opt=1.0)
    local_search = LocalSearch(ctx)
    network_class = classify_network(ctx.num_pipes)

    worker = ProbedWorker(ctx, KickStrategies(ctx, local_search), local_search,
                          0, 1, max_sims, beam_width_for(network_class), [])
    worker.run(float("inf"), float("inf"), None)
    return worker, ctx


def summarise(name, num_pipes, samples):
    if not samples:
        return None

    all_gaps = [s["gap_max"] for s in samples]
    target_gaps = [s["gap_target"] for s in samples if s["gap_target"] is not None]
    pool_sizes = [s["pool_size"] for s in samples]
    temperatures = [s["temperature"] for s in samples]

    row = {
        "network": name,
        "pipes": num_pipes,
        "kicks": len(samples),
        "pool_size_mean": round(statistics.mean(pool_sizes), 2),
        "pool_size_max": max(pool_sizes),
        "T_mean": round(statistics.mean(temperatures), 3),
        "T_max": max(temperatures),
        "gap_worst_member": round(max(all_gaps), 4),
        "gap_worst_mean": round(statistics.mean(all_gaps), 4),
        "gap_target_max": round(max(target_gaps), 4) if target_gaps else None,
        "gap_target_mean": round(statistics.mean(target_gaps), 4) if target_gaps else None,
    }
    for threshold in GAP_THRESHOLDS:
        over = sum(1 for gap in all_gaps if gap > threshold)
        row[f"kicks_with_member_over_{int(threshold * 100)}pct"] = over
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--hmin", type=float, required=True)
    parser.add_argument("--units", default="mm")
    parser.add_argument("--max_sims", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="experiments/pool_gaps")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.inp))[0]
    worker, ctx = run_probe(args.inp, args.costs, args.hmin, args.max_sims,
                            args.seed, args.units)

    os.makedirs(args.out, exist_ok=True)
    detail_path = os.path.join(args.out, f"{name}_kicks.csv")
    if worker.samples:
        with open(detail_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(worker.samples[0]))
            writer.writeheader()
            writer.writerows(worker.samples)

    row = summarise(name, ctx.num_pipes, worker.samples)
    print("\n=== POOL GAP PROBE ===")
    if row is None:
        print(f"{name}: жодного кіка з непорожнім пулом (бюджет замалий)")
        return
    for key, value in row.items():
        print(f"  {key:34s} {value}")
    print(f"\n  деталі: {detail_path}")


if __name__ == "__main__":
    main()
