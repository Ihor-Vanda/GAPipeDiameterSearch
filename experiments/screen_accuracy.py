"""Чи можна відсіювати кандидатів без запуску EPANET — і чи можна це
відкалібрувати самому алгоритму на льоту.

Ідея відсіву. Зміна діаметра однієї труби змінює її втрату напору у
відомий бік: при незмінній витраті, за Хазеном-Вільямсом,

    h ~ 1 / d^4.87   =>   h_нове = h_старе · (d_старе / d_нове)^4.87

Якщо труба лежить на шляху від джерела до критичного вузла, приріст
тиску в критичному вузлі в першому наближенні дорівнює `−Δh` цієї труби;
якщо не лежить — нулю. Отже знак і порядок зміни `p_min` можна оцінити з
уже обчислених втрат напору (`get_cached_heuristics`) і домінантного
шляху (`get_dominant_path`) **без розв'язання гідравліки**.

Наближення грубе рівно в тій мірі, в якій у мережі перерозподіляються
витрати, тобто залежить від кільцевості. Тому тут воно міряється на
мережах із різко різною кількістю контурів (Ханої 3, Балерма 8, сітка
24×24 — 529).

**Головне питання не в самій точності, а в тому, чи може алгоритм
оцінити її сам.** Якщо точність, виміряна на перших 20–50 кандидатах,
передбачає точність на решті, то поріг довіри до відсіву не треба
зашивати за класом мережі: солвер відкалібрує його на тому екземплярі,
який йому дали, хай там 30 труб чи 4000.

    python experiments/screen_accuracy.py --networks InputData/Hanoi/Hanoi.inp:in:30 \
        InputData/Balerma/Balerma.inp:mm:20 --sizes 24
"""

import argparse
import csv
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import networkx as nx  # noqa: E402

from analytical_solver.context import SolverContext  # noqa: E402
from analytical_solver.local_search import LocalSearch  # noqa: E402
from config import SolverConfig  # noqa: E402
from cost_table import load_cost_table  # noqa: E402
from make_grid_network import build  # noqa: E402
from water_sim import WaterSimulator  # noqa: E402

#: Показник Хазена-Вільямса для діаметра.
DIAMETER_EXPONENT = 4.87
BALERMA_COSTS = os.path.join("InputData", "Balerma", "costs.csv")


def predict_delta_p(ctx, indices, unit_losses, path_pipes, pipe_idx, new_d_idx):
    """Оцінка зміни `p_min` від зміни діаметра однієї труби, без симуляції."""
    if pipe_idx not in path_pipes:
        return 0.0

    headloss = unit_losses[pipe_idx] * ctx.lengths[pipe_idx]
    old_d = ctx.diameters[indices[pipe_idx]]
    new_d = ctx.diameters[new_d_idx]
    return headloss * (1.0 - (old_d / new_d) ** DIAMETER_EXPONENT)


def node_paths(ctx):
    """Для кожного вузла — множина труб на шляху від джерела до нього.

    Обчислюється один раз із топології, без гідравліки. Це і є те, чого
    бракувало наївній оцінці: `p_min` — мінімум по **всіх** вузлах, тому
    стежити лише за шляхом до старого критичного вузла недостатньо.
    """
    sources = ctx.simulator.sources or []
    if not sources:
        return {}

    source = sources[0]
    pipes_between = {}
    for node in ctx.base_G_flow.nodes():
        try:
            route = nx.shortest_path(ctx.base_G_flow, source, node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        pipes = set()
        for a, b in zip(route, route[1:]):
            shared = set(ctx.node_to_pipes.get(a, [])) & set(ctx.node_to_pipes.get(b, []))
            pipes.update(shared)
        pipes_between[node] = pipes
    return pipes_between


def node_pressures(ctx, indices):
    """Тиски в усіх вузлах-відборах для заданого рішення."""
    simulator = ctx.simulator
    simulator._apply_pattern(indices)
    if not simulator._run_simulation_core():
        return None
    return {simulator.c_idx_to_node_name[c_idx]:
            simulator.api.ENgetnodevalue(c_idx, 11) * simulator.press_si_mult
            for c_idx in simulator.junction_c_indices}


def predict_delta_p_all_nodes(ctx, indices, unit_losses, paths, pressures,
                              pipe_idx, new_d_idx):
    """Оцінка нового `p_min` через усі вузли, а не лише критичний."""
    headloss = unit_losses[pipe_idx] * ctx.lengths[pipe_idx]
    old_d = ctx.diameters[indices[pipe_idx]]
    new_d = ctx.diameters[new_d_idx]
    delta_h = headloss * ((old_d / new_d) ** DIAMETER_EXPONENT - 1.0)

    worst = float("inf")
    for node, pressure in pressures.items():
        affected = pipe_idx in paths.get(node, ())
        worst = min(worst, pressure - delta_h if affected else pressure)
    return worst


def spearman(xs, ys):
    """Кореляція рангів без scipy."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                result[order[k]] = average
            i = j + 1
        return result

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def evaluate(ctx, local_search, name, h_min, sample_limit, mode="paths"):
    """Порівняти оцінку з фактом на зійшовшомуся розв'язку."""
    graph = ctx.simulator.wn.get_graph() if hasattr(ctx.simulator, "wn") else None
    loops = (graph.number_of_edges() - graph.number_of_nodes() + 1) if graph else None

    start = [ctx.max_d_idx] * ctx.num_pipes
    dyn = ctx.get_cached_stats(start).cost * 0.001
    random.seed(42)
    solution = local_search.gradient_squeeze(start, max_passes=None, quick_mode=False,
                                             dyn_bonus=dyn, min_rel_improvement=0.0)

    base = ctx.get_cached_stats(solution)
    unit_losses = ctx.get_cached_heuristics(solution)
    path_pipes = set(ctx.get_dominant_path(solution, base.crit_node)[0] or [])

    candidates = [i for i in range(ctx.num_pipes) if solution[i] > 0]
    random.shuffle(candidates)
    candidates = candidates[:sample_limit]

    paths = node_paths(ctx)
    pressures = node_pressures(ctx, solution) or {}

    predicted, actual = [], []
    for pipe_idx in candidates:
        new_d_idx = solution[pipe_idx] - 1          # крок вниз: саме він ризикує тиском
        if mode == "dominant":
            estimate = predict_delta_p(ctx, solution, unit_losses, path_pipes,
                                       pipe_idx, new_d_idx)
        else:
            estimate = (predict_delta_p_all_nodes(ctx, solution, unit_losses, paths,
                                                  pressures, pipe_idx, new_d_idx)
                        - base.p_min)
        predicted.append(estimate)
        trial = list(solution)
        trial[pipe_idx] = new_d_idx
        actual.append(ctx.get_cached_stats(trial).p_min - base.p_min)

    # Відсів: кандидат відкидається, якщо оцінка обіцяє падіння тиску нижче запасу.
    surplus = base.p_min - h_min
    rejected = [p < -surplus for p in predicted]
    truly_infeasible = [a < -surplus for a in actual]

    true_reject = sum(1 for r, t in zip(rejected, truly_infeasible) if r and t)
    false_reject = sum(1 for r, t in zip(rejected, truly_infeasible) if r and not t)
    missed = sum(1 for r, t in zip(rejected, truly_infeasible) if not r and t)
    kept_ok = sum(1 for r, t in zip(rejected, truly_infeasible) if not r and not t)

    on_path = sum(1 for i in candidates if i in path_pipes)
    correct_sign = sum(1 for p, a in zip(predicted, actual)
                       if (p >= 0) == (a >= 0))

    # Чи можна оцінити точність на льоту: рахуємо ту саму частку збігів
    # знаку на перших 20 і на перших 50 кандидатах.
    def sign_share(limit):
        subset = list(zip(predicted, actual))[:limit]
        if not subset:
            return None
        return sum(1 for p, a in subset if (p >= 0) == (a >= 0)) / len(subset)

    return {
        "network": name,
        "mode": mode,
        "pipes": ctx.num_pipes,
        "loops": loops,
        "candidates": len(candidates),
        "on_dominant_path": on_path,
        "pressure_surplus_m": round(surplus, 3),
        "spearman": (round(spearman(predicted, actual), 3)
                     if spearman(predicted, actual) is not None else None),
        "sign_accuracy": round(correct_sign / max(1, len(candidates)), 3),
        "sign_accuracy_first20": (round(sign_share(20), 3) if sign_share(20) else None),
        "sign_accuracy_first50": (round(sign_share(50), 3) if sign_share(50) else None),
        "screen_rejected": sum(rejected),
        "reject_correct": true_reject,
        "reject_wrong": false_reject,
        "missed_infeasible": missed,
        "kept_feasible": kept_ok,
    }


def load(inp_file, cost_file, h_min, units):
    config = SolverConfig(cost_file=os.path.abspath(cost_file), h_min=h_min,
                      unit_system=units, run_mode="analytical", v_opt=1.0)
    load_cost_table(config)
    temp_dir = os.path.abspath(os.path.join("_temp_sim_data", "screen"))
    os.makedirs(temp_dir, exist_ok=True)
    simulator = WaterSimulator(os.path.abspath(inp_file), config, temp_dir=temp_dir)
    ctx = SolverContext(simulator, config.diameters_m, v_opt=1.0)
    return ctx, LocalSearch(ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks", nargs="*", default=[])
    parser.add_argument("--sizes", type=int, nargs="*", default=[])
    parser.add_argument("--sample", type=int, default=200,
                        help="Скільки кандидатів перевіряти симуляцією")
    parser.add_argument("--out", default="experiments/screen_accuracy.csv")
    args = parser.parse_args()

    jobs = []
    for spec in args.networks:
        path, units, h_min = spec.split(":")
        jobs.append((path, os.path.join(os.path.dirname(path), "costs.csv"),
                     float(h_min), units))
    for side in args.sizes:
        text, _, _ = build(side, side)
        folder = os.path.join("_temp_sim_data", "nets", f"Grid{side}x{side}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"Grid{side}x{side}.inp")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        jobs.append((path, BALERMA_COSTS, 20.0, "mm"))

    rows = []
    for inp_file, cost_file, h_min, units in jobs:
        name = os.path.splitext(os.path.basename(inp_file))[0]
        ctx, local_search = load(inp_file, cost_file, h_min, units)
        for mode in ("dominant", "paths"):
            row = evaluate(ctx, local_search, name, h_min, args.sample, mode=mode)
            rows.append(row)
            print(f"  {row['network']:<12} труб={row['pipes']:<5} "
                  f"контурів={row['loops']:<5} {row['mode']:<9} "
                  f"| знак {100*row['sign_accuracy']:>5.1f} % "
                  f"(20: {100*(row['sign_accuracy_first20'] or 0):>5.1f} %, "
                  f"50: {100*(row['sign_accuracy_first50'] or 0):>5.1f} %) "
                  f"| Спірмен {str(row['spearman']):>6} "
                  f"| відсів {row['screen_rejected']:>4}: вірно {row['reject_correct']:>4}, "
                  f"хибно {row['reject_wrong']:>4}, пропущено "
                  f"{row['missed_infeasible']:>4}", flush=True)

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  {args.out}")


if __name__ == "__main__":
    main()
