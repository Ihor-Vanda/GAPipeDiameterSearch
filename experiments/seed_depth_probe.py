"""Скільки проходів справді потрібно спуску насіння.

`make_diverse_seeds` полірує кожне зерно `gradient_squeeze(max_passes=12)`.
Замір показав, що саме ці три виклики і є ціною насіння: 4 190 симуляцій
за виклик на Балермі проти 168 на Ханої. Питання те саме, що й на етапі
25, але про інше місце: **чи окупаються останні проходи**.

Стенд бере ту саму стартову точку, що й `make_diverse_seeds` (рішення за
цільовою швидкістю, за потреби вилікуване), і спускає з різною стелею
проходів, щоразу з **холодного кеша** — інакше глибший варіант виглядав
би дешевшим, ніж він є.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.config import SolverConfig  # noqa: E402
from pipeopt.hydraulics.water_sim import WaterSimulator  # noqa: E402
from pipeopt.reporting.cost_table import load_cost_table  # noqa: E402
from pipeopt.solver.context import SolverContext  # noqa: E402
from pipeopt.solver.local_search import LocalSearch  # noqa: E402
from pipeopt.solver.seeds import SeedFactory  # noqa: E402

NETS = {"Hanoi": ("Hanoi.inp", "in", 30.0),
        "Foss": ("Foss.inp", "mm", 40.0),
        "Balerma": ("Balerma.inp", "mm", 20.0),
        "Grid16x16": ("Grid16x16.inp", "mm", 20.0),
        "Grid24x24": ("Grid24x24.inp", "mm", 20.0)}
DEPTHS = (1, 2, 3, 4, 6, 8, 12)


def fresh(net):
    inp, units, h_min = NETS[net]
    config = SolverConfig(cost_file=os.path.join(ROOT, "InputData", net, "costs.csv"),
                          unit_system=units, h_min=h_min)
    load_cost_table(config, silent=True)
    temp = os.path.abspath(os.path.join("_temp_sim_data", f"seeddepth_{net}"))
    os.makedirs(temp, exist_ok=True)
    simulator = WaterSimulator(os.path.join(ROOT, "InputData", net, inp),
                               config, temp_dir=temp)
    ctx = SolverContext(simulator, config.diameters_m, v_opt=1.0)
    ls = LocalSearch(ctx)
    return ctx, ls, SeedFactory(ctx, ls)


def start_points(ctx, ls, seeder):
    """Ті самі стартові точки, що їх будує `make_diverse_seeds`."""
    points = []
    for v in (1.0, 1.2, 0.8):
        idx_sol = [ctx.max_d_idx] * ctx.num_pipes
        for _ in range(10):
            real = [ctx.diameters[i] for i in idx_sol]
            flows, _ = ctx.simulator.get_hydraulic_state(real)
            new_idx = [seeder.calculate_ideal_d(q, v) for q in flows]
            if new_idx == idx_sol:
                break
            idx_sol = new_idx
        _, p, _, _ = ctx.get_cached_stats(idx_sol)
        if p < ctx.simulator.config.h_min:
            idx_sol, ok, _ = ls.heal_network(idx_sol, set())
            if not ok:
                continue
        points.append((v, list(idx_sol)))
    return points


def main():
    net = os.environ.get("NET", "Hanoi")
    ctx, ls, seeder = fresh(net)
    points = start_points(ctx, ls, seeder)
    ctx.simulator.close()
    print(f"{net}: {len(points)} стартових точок")

    rows = []
    for v, start in points:
        line = []
        for depth in DEPTHS:
            ctx, ls, _ = fresh(net)        # холодний кеш на кожну глибину
            # Те саме зерно перед кожною глибиною: спуск перемішує
            # труби, і без цього мілкий варіант не був би префіксом
            # глибокого — числа стрибали б немонотонно від RNG, а не
            # від глибини (перша версія стенда саме це й показала).
            random.seed(20260917)
            before = ctx.sim_count
            result = ls.gradient_squeeze(start, max_passes=depth, quick_mode=True)
            cost, _, _, _ = ctx.get_cached_stats(result)
            line.append((depth, cost, ctx.sim_count - before))
            ctx.simulator.close()
        best = line[-1][1]
        print(f"  v={v}:  " + "  ".join(
            f"{d}п {c/1e6:.4f}M/{s:,}" for d, c, s in line))
        print("        відставання від 12 проходів: " + "  ".join(
            f"{d}п {(c - best)/best:+.3%}" for d, c, s in line))
        rows.append({"v": v, "line": line})

    with open(os.environ.get("OUT", f"/tmp/claude-0/seeddepth_{net}.json"), "w") as fh:
        json.dump({"net": net, "rows": rows}, fh)


main()
