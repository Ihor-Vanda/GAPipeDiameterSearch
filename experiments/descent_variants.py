"""Чи можна зробити градієнтний спуск дешевшим, не втративши якість.

Замір §22 показав: основна витрата симуляцій — `gradient_squeeze`, який
щопроходу перебирає **всі** труби, пробуючи для кожної −1 і +1. У
зійшовшомуся стані поліпшуваних труб **нуль із 454**, але доказ цього
коштує Θ(n) симуляцій і робиться заново при кожному виклику, хоча між
викликами змінилось 1–2 труби.

Тут порівнюються чотири способи вибирати, які труби взагалі перебирати:

* `baseline`  — усі труби щопроходу (як зараз);
* `dontlook`  — «біти не-дивись»: труба, для якої хід не вдався, більше не
                перевіряється, доки не змінилась вона сама або труба з її
                гідравлічного сусідства (класика локального пошуку для
                2-opt, Bentley 1992);
* `topk`      — лише труби найвищого впливу (`get_high_impact_pipes`, який
                уже є в проєкті й уже застосовується до мутацій, але не до
                спуску);
* `both`      — обидва обмеження разом.

Міряється однаково для всіх: **скільки симуляцій витрачено** і **яка
вартість отримана** з того самого стартового рішення. Якість —
це вартість, а не схожість векторів: інший локальний оптимум тієї ж
вартості цілком годиться.

Перший варіант — контрольний: він реалізує ту саму логіку, що
`LocalSearch.gradient_squeeze`, і тест `--verify` перевіряє, що на
однаковому seed вони дають однаковий результат. Без цієї перевірки
порівняння нічого не варте.

    python experiments/descent_variants.py --networks InputData/Balerma/Balerma.inp:mm:20
    python experiments/descent_variants.py --sizes 16 24 --starts max seeded
"""

import argparse
import csv
import os
import random
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from analytical_solver.context import SolverContext  # noqa: E402
from analytical_solver.local_search import LocalSearch  # noqa: E402
from config import SolverConfig  # noqa: E402
from cost_table import load_cost_table  # noqa: E402
from make_grid_network import build  # noqa: E402
from water_sim import WaterSimulator  # noqa: E402

VARIANTS = ("baseline", "dontlook", "topk", "both")
BALERMA_COSTS = os.path.join("InputData", "Balerma", "costs.csv")


def pipe_neighbours(ctx):
    """Труби, що мають спільний вузол: гідравлічне сусідство першого порядку."""
    neighbours = [set() for _ in range(ctx.num_pipes)]
    for pipes in ctx.node_to_pipes.values():
        for a in pipes:
            neighbours[a].update(pipes)
    for a in range(ctx.num_pipes):
        neighbours[a].discard(a)
    return [sorted(s) for s in neighbours]


def descent(ctx, local_search, start, dyn_bonus, variant, top_k=None,
            max_passes=None):
    """Градієнтний спуск із заданим правилом вибору труб.

    Логіка руху — та сама, що в `LocalSearch.gradient_squeeze` у
    неквадратичному (`quick_mode=False`) режимі: для труби пробуються
    −1 і +1, приймається найкращий допустимий хід, що поліпшує оцінку.
    Відрізняється лише те, **які** труби перебираються.
    """
    current = list(start)
    stats = ctx.get_cached_stats(current)
    cost, p_min = stats.cost, stats.p_min
    best_score = ctx.score(cost, p_min, dyn_bonus)

    use_dontlook = variant in ("dontlook", "both")
    use_topk = variant in ("topk", "both")

    all_pipes = set(range(ctx.num_pipes))
    neighbours = pipe_neighbours(ctx) if use_dontlook else None
    # «Брудні» труби — ті, які ще варто перевірити.
    dirty = set(all_pipes)

    sims_before = ctx.sim_count
    passes = 0
    improved = True

    while improved:
        improved = False
        passes += 1
        if max_passes and passes > max_passes:
            break

        # Вплив труби залежить від поточного рішення, тому топ-k
        # перераховується щопроходу: інакше спуск назавжди замикається
        # на трубах, які були цікаві лише на старті.
        allowed = (set(local_search.get_high_impact_pipes(current, top_k))
                   if use_topk else all_pipes)

        scan = sorted(dirty & allowed) if use_dontlook else sorted(allowed)
        if not scan:
            break
        random.shuffle(scan)

        for idx in scan:
            current_d = current[idx]
            best_move = None
            move_score, move_cost, move_p = best_score, cost, p_min

            for delta in (-1, +1):
                new_d = current_d + delta
                if not 0 <= new_d <= ctx.max_d_idx:
                    continue
                trial = list(current)
                trial[idx] = new_d
                trial_stats = ctx.get_cached_stats(trial)
                if trial_stats.p_min < ctx.h_min:
                    continue
                score = ctx.score(trial_stats.cost, trial_stats.p_min, dyn_bonus)
                if score < move_score:
                    best_move = trial
                    move_score, move_cost, move_p = (score, trial_stats.cost,
                                                     trial_stats.p_min)

            if best_move is not None:
                current = best_move
                best_score, cost, p_min = move_score, move_cost, move_p
                improved = True
                if use_dontlook:
                    # Змінилась ця труба -> вона сама та її сусіди знову цікаві.
                    dirty.add(idx)
                    dirty.update(neighbours[idx])
            elif use_dontlook:
                dirty.discard(idx)

    return {
        "solution": current,
        "cost": cost,
        "p_min": p_min,
        "sims": ctx.sim_count - sims_before,
        "passes": passes,
    }


def reset_cache(ctx):
    """Однаковий холодний старт для кожного варіанту."""
    from analytical_solver.cache import LRUCache
    ctx.sim_cache = LRUCache(ctx.sim_cache.maxsize)
    ctx.heuristic_cache = LRUCache(ctx.heuristic_cache.maxsize)


def load_network(inp_file, cost_file, h_min, units):
    config = SolverConfig(cost_file=os.path.abspath(cost_file), h_min=h_min,
                      unit_system=units, run_mode="analytical", v_opt=1.0)
    load_cost_table(config)
    temp_dir = os.path.abspath(os.path.join("_temp_sim_data", "descent"))
    os.makedirs(temp_dir, exist_ok=True)
    simulator = WaterSimulator(os.path.abspath(inp_file), config, temp_dir=temp_dir)
    ctx = SolverContext(simulator, config.diameters_m, v_opt=1.0)
    return ctx, LocalSearch(ctx)


def starting_point(ctx, local_search, kind, perturb=2, seed=42):
    """Стартове рішення для спуску.

    `max`       — усі максимальні (найдальший можливий старт);
    `seeded`    — гідравлічна нерухома точка, як у насінні;
    `perturbed` — **сценарій, який справді трапляється в алгоритмі**:
                  зійшовшийся розв'язок, у якому кік змінив кілька труб.
                  Саме тут спуск запускається найчастіше, і саме тут
                  економія має сенс.
    """
    all_max = [ctx.max_d_idx] * ctx.num_pipes
    if kind == "max":
        return all_max

    if kind == "perturbed":
        base = starting_point(ctx, local_search, "seeded")
        dyn = ctx.get_cached_stats(base).cost * 0.001
        random.seed(seed)
        converged = descent(ctx, local_search, base, dyn, "baseline")["solution"]

        # Кік підіймає кілька труб на крок вище: далі спуск мусить
        # довести, що решта мережі лишилась оптимальною.
        perturbed = list(converged)
        for idx in random.sample(range(ctx.num_pipes), min(perturb, ctx.num_pipes)):
            perturbed[idx] = min(ctx.max_d_idx, perturbed[idx] + 1)
        return perturbed

    # Гідравлічно осмислений старт: та сама нерухома точка, що в
    # `SeedFactory.make_diverse_seeds`, але без поліровки — інакше
    # порівнювати спуски не було б з чого.
    from analytical_solver.orchestrator import SeedFactory
    seeder = SeedFactory(ctx, local_search, 1, 5)
    solution = all_max
    for _ in range(10):
        real = [ctx.diameters[i] for i in solution]
        flows, _ = ctx.simulator.get_hydraulic_state(real)
        nxt = [seeder.calculate_ideal_d(q, 1.0) for q in flows]
        if nxt == solution:
            break
        solution = nxt
    if ctx.get_cached_stats(solution).p_min < ctx.h_min:
        healed, ok, _ = local_search.heal_network(solution, set())
        if ok:
            solution = healed
    return solution


def verify_baseline(ctx, local_search, start, seed):
    """Контрольний варіант мусить збігатися з наявним `gradient_squeeze`."""
    dyn = ctx.get_cached_stats(start).cost * 0.001

    random.seed(seed)
    mine = descent(ctx, local_search, start, dyn, "baseline")

    random.seed(seed)
    theirs = local_search.gradient_squeeze(start, max_passes=None, quick_mode=False,
                                           dyn_bonus=dyn, min_rel_improvement=0.0)
    theirs_cost = ctx.get_cached_stats(theirs).cost
    return mine["cost"], theirs_cost, mine["solution"] == list(theirs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks", nargs="*", default=[])
    parser.add_argument("--sizes", type=int, nargs="*", default=[])
    parser.add_argument("--starts", nargs="*", default=["max", "seeded"])
    parser.add_argument("--top_k_divisor", type=int, default=5,
                        help="k = num_pipes // цей дільник (як у мутаціях)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturb", type=int, default=2,
                        help="Скільки труб зсуває «кік» для старту `perturbed`")
    parser.add_argument("--verify", action="store_true",
                        help="Перевірити, що контрольний варіант збігається з кодом проєкту")
    parser.add_argument("--out", default="experiments/descent_variants.csv")
    args = parser.parse_args()

    jobs = []
    for spec in args.networks:
        path, units, h_min = spec.split(":")
        costs = os.path.join(os.path.dirname(path), "costs.csv")
        jobs.append((path, costs, float(h_min), units))
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
        ctx, local_search = load_network(inp_file, cost_file, h_min, units)
        top_k = max(10, ctx.num_pipes // args.top_k_divisor)

        for start_kind in args.starts:
            start = starting_point(ctx, local_search, start_kind,
                                   perturb=args.perturb, seed=args.seed)
            dyn = ctx.get_cached_stats(start).cost * 0.001
            start_cost = ctx.get_cached_stats(start).cost

            if args.verify:
                mine, theirs, same = verify_baseline(ctx, local_search, start, args.seed)
                print(f"  [verify] {name}/{start_kind}: контрольний {mine/1e6:.4f} M$, "
                      f"проєктний {theirs/1e6:.4f} M$, вектори {'збігаються' if same else 'РІЗНІ'}")

            reference = None
            for variant in VARIANTS:
                random.seed(args.seed)
                # Кеш обнуляється перед кожним варіантом: інакше той, що
                # біжить другим, дістає чужі результати безкоштовно.
                reset_cache(ctx)
                started = time.perf_counter()
                result = descent(ctx, local_search, start, dyn, variant, top_k=top_k)
                elapsed = time.perf_counter() - started
                if reference is None:
                    reference = result

                row = {
                    "network": name, "pipes": ctx.num_pipes, "start": start_kind,
                    "variant": variant, "top_k": top_k if "topk" in variant or variant == "both" else "",
                    "start_cost": round(start_cost, 1),
                    "cost": round(result["cost"], 1),
                    "cost_vs_baseline_pct": round(100 * (result["cost"] - reference["cost"])
                                                  / reference["cost"], 3),
                    "sims": result["sims"],
                    "sims_vs_baseline": round(result["sims"] / max(1, reference["sims"]), 3),
                    "passes": result["passes"],
                    "p_min": round(result["p_min"], 2),
                    "seconds": round(elapsed, 2),
                }
                rows.append(row)
                print(f"  {name:<12} {start_kind:<7} {variant:<9} "
                      f"{result['cost']/1e6:>9.4f} M$ ({row['cost_vs_baseline_pct']:+7.3f} %) | "
                      f"{result['sims']:>7} сим (×{row['sims_vs_baseline']:.3f}) | "
                      f"{result['passes']:>3} прох. | {elapsed:>6.1f} с", flush=True)

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  {args.out}")


if __name__ == "__main__":
    main()
