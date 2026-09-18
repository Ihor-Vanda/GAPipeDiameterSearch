"""Програти записаних кандидатів із різною глибиною поліровки.

Питання, на яке відповідає стенд: **чи змінюють проходи 2 і 3 те, що
промінь справді вибирає.** Промінь ранжує кандидатів і бере топ-k; якщо
топ-k однаковий при одному проході й при трьох, то два зайві проходи
куплені даремно — і це можна сказати числом, а не на око.

Чесність заміру тримається на трьох речах:

* **свіжий контекст на кожну глибину** — кеш симуляцій холодний, тому
  ціна проходів порівнюється чесно, а не «другий замір дешевший, бо
  перший усе прогрів»;
* **однакове зерно перед кожним кандидатом** — `gradient_squeeze`
  перемішує труби, тому без цього один прохід не був би префіксом трьох;
* **кандидати взяті зі справжнього прогону**, а не згенеровані: саме
  такий набір ходів пропонує `_generate_mutations`, і саме на ньому
  питання має сенс.

Стенд пише поканд��датні оцінки й ціни у файл: правила зупинки далі
програються офлайн, без жодної симуляції, і всі вони порівнюються на
**тих самих** числах.
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

NETS = {"Hanoi": ("Hanoi.inp", "in", 30.0),
        "Balerma": ("Balerma.inp", "mm", 20.0),
        "Foss": ("Foss.inp", "mm", 40.0)}
DEPTHS = (1, 2, 3)


def fresh(net):
    inp, units, h_min = NETS[net]
    config = SolverConfig(cost_file=os.path.join(ROOT, "InputData", net, "costs.csv"),
                          unit_system=units, h_min=h_min)
    load_cost_table(config, silent=True)
    temp = os.path.abspath(os.path.join("_temp_sim_data", f"replay_{net}"))
    os.makedirs(temp, exist_ok=True)
    simulator = WaterSimulator(os.path.join(ROOT, "InputData", net, inp),
                              config, temp_dir=temp)
    ctx = SolverContext(simulator, config.diameters_m, v_opt=1.0)
    return ctx, LocalSearch(ctx)


def score_all(net, candidates, depth):
    """Оцінити кожного кандидата з заданою глибиною; повернути оцінки й ціни."""
    ctx, ls = fresh(net)
    scores, costs = [], []
    original = ls.gradient_squeeze

    def squeeze(*args, **kwargs):
        if kwargs.get("max_passes") == 3:
            kwargs["max_passes"] = depth
        return original(*args, **kwargs)

    ls.gradient_squeeze = squeeze
    for number, candidate in enumerate(candidates):
        random.seed(number)          # той самий порядок труб на всіх глибинах
        before = ctx.sim_count
        score, _, _ = ls.evaluate_candidate(
            candidate["base"], candidate["pipes"], candidate["mode"],
            candidate["dyn_bonus"])
        scores.append(score if score != float("inf") else None)
        costs.append(ctx.sim_count - before)
    ctx.simulator.close()
    return scores, costs


def main():
    net = os.environ.get("NET", "Hanoi")
    data = json.load(open(os.environ["IN"]))
    candidates = data["candidates"]

    out = {"net": net,
           "beam_width": data["beam_width"],
           "pipes": data["pipes"],
           "round": [c["round"] for c in candidates],
           "scores": {}, "costs": {}}
    for depth in DEPTHS:
        scores, costs = score_all(net, candidates, depth)
        out["scores"][str(depth)] = scores
        out["costs"][str(depth)] = costs
        print(f"  глибина {depth}: {sum(costs):,} симуляцій")

    with open(os.environ["OUT"], "w") as fh:
        json.dump(out, fh)


main()
