"""Куди йде час і симуляції на стадії насіння.

Виникло з етапу 25 і зі спростування власної здогадки про те, що
насіння з'їдає прогін на великій мережі. Стенд викликає `SeedFactory`
**напряму**, без прогону, і рахує окремо кожен її крок, а всередині —
`heal_network` і `gradient_squeeze`. Так видно не «насіння дороге», а
**який саме виклик** дорогий.

NET — мережа, CAP — стеля на один крок у секундах (крок, що її перевищив,
позначається як обірваний, і стенд іде далі).
"""
import json
import os
import sys
import time

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

CALLS = []


def instrument(ls, ctx):
    """Порахувати кожен виклик лікування й спуску окремо."""
    heal, squeeze = ls.heal_network, ls.gradient_squeeze

    def wrap(name, func):
        def inner(*args, **kwargs):
            before, start = ctx.sim_count, time.time()
            try:
                return func(*args, **kwargs)
            finally:
                CALLS.append({"what": name,
                              "passes": kwargs.get("max_passes"),
                              "sims": ctx.sim_count - before,
                              "secs": round(time.time() - start, 2)})
        return inner

    ls.heal_network = wrap("heal", heal)
    ls.gradient_squeeze = wrap("squeeze", squeeze)


def main():
    net = os.environ.get("NET", "Hanoi")
    inp, units, h_min = NETS[net]
    config = SolverConfig(cost_file=os.path.join(ROOT, "InputData", net, "costs.csv"),
                          unit_system=units, h_min=h_min)
    load_cost_table(config, silent=True)
    temp = os.path.abspath(os.path.join("_temp_sim_data", f"seedcost_{net}"))
    os.makedirs(temp, exist_ok=True)
    simulator = WaterSimulator(os.path.join(ROOT, "InputData", net, inp),
                               config, temp_dir=temp)
    ctx = SolverContext(simulator, config.diameters_m, v_opt=1.0)
    ls = LocalSearch(ctx)
    instrument(ls, ctx)
    seeder = SeedFactory(ctx, ls)

    import random
    random.seed(42)
    print(f"{net}: {ctx.num_pipes} труб")
    for name, call in (("make_diverse_seeds", seeder.make_diverse_seeds),
                       ("make_reserve_pool", lambda: seeder.make_reserve_pool(size=4))):
        before, start = ctx.sim_count, time.time()
        mark = len(CALLS)
        call()
        print(f"  {name}: {ctx.sim_count - before:,} симуляцій, "
              f"{time.time() - start:.1f} с, викликів {len(CALLS) - mark}")

    by = {}
    for record in CALLS:
        key = (record["what"], record["passes"])
        entry = by.setdefault(key, {"n": 0, "sims": 0, "secs": 0.0, "max": 0})
        entry["n"] += 1
        entry["sims"] += record["sims"]
        entry["secs"] += record["secs"]
        entry["max"] = max(entry["max"], record["sims"])
    print("  розклад за викликами:")
    for (what, passes), entry in sorted(by.items(), key=lambda kv: -kv[1]["sims"]):
        label = f"{what}(max_passes={passes})" if what == "squeeze" else what
        print(f"    {label:<28} ×{entry['n']:<3} {entry['sims']:>8,} сим  "
              f"{entry['secs']:>6.1f} с  найдорожчий {entry['max']:,}")

    with open(os.environ.get("OUT", f"/tmp/claude-0/seedcost_{net}.json"), "w") as fh:
        json.dump({"net": net, "pipes": ctx.num_pipes, "calls": CALLS}, fh)
    simulator.close()


main()
