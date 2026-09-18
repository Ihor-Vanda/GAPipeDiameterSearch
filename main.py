import argparse
import json
import multiprocessing
import os
import sys
from datetime import datetime

from pipeopt.solver.worker import claim_temp_root, clean_temp_root
from pipeopt.console import DualLogger, silence_warnings
from pipeopt.runner import OptimizationRunner, RunRequest, setup_run_directory

silence_warnings()


def build_parser():
    """Аргументи CLI.

    Винесено з `main()` на етапі 18: доти розбір аргументів не мав жодного
    тесту, бо перевірити його можна було лише запустивши прогін цілком.
    """
    parser = argparse.ArgumentParser(description="Analytical Solver for WDN")

    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (overrides other args)")
    parser.add_argument("--max_sims", type=int, default=None, help="Global simulation budget (e.g., 15000000)")

    parser.add_argument("--inp", type=str, default="InputData/Hanoi/Hanoi.inp", help="Path to EPANET .inp file")
    parser.add_argument("--costs", type=str, default="InputData/Hanoi/costs.csv", help="Path to cost configuration JSON")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--hmin", type=float, default=30.0, help="Minimum allowable head (pressure)")
    parser.add_argument("--units", type=str, choices=["in", "mm"], default="mm", help="Units for pipe diameters")
    parser.add_argument("--cores", type=int, default=0, help="Number of CPU cores (0 = all)")

    parser.add_argument("--run_mode", type=str, choices=["fast_analytical", "analytical"], default="analytical", help="Mode: fast analytical or analytical")
    parser.add_argument("--v_opt", type=float, default=1.0, help="Optimal velocity (m/s) for analytical solver")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed of the main process. Omit for a non-deterministic run. "
                             "Combine with --sequential for a bit-exact reproducible run.")
    parser.add_argument("--sequential", action="store_true",
                        help="Run a single island in-process, without multiprocessing.Pool, "
                             "and stop strictly on the simulation budget. Slower, but the only "
                             "mode that is bit-exact reproducible for a given --seed. "
                             "Use it for baselines and A/B comparisons.")
    parser.add_argument("--time_limit", type=float, default=None,
                        help="Wall-clock limit per run in seconds; 0 = unlimited (stop only on "
                             "--max_sims). Default: derived from the calibrated simulator speed, "
                             "which makes the stopping point machine- and load-dependent.")

    return parser


def apply_config_file(args, argv=None):
    """Накласти JSON-конфіг на аргументи — не перекриваючи того, що ввели руками.

    Правило саме таке: значення з файлу програє явному прапорцю в
    командному рядку. Інакше конфіг для серії прогонів мовчки ігнорував би
    `--seed`, яким цю серію й розрізняють.

    Невідомі ключі у файлі пропускаються: конфіги пишуться руками, і
    падати через зайвий рядок було б гірше, ніж його не помітити.
    """
    if not args.config or not os.path.exists(args.config):
        return args

    argv = sys.argv if argv is None else argv
    explicit = {arg.lstrip('-').split('=')[0] for arg in argv if arg.startswith('--')}

    with open(args.config, encoding='utf-8') as f:
        config_data = json.load(f)
    for key, value in config_data.items():
        if key not in explicit and hasattr(args, key):
            setattr(args, key, value)
    return args


def request_from_args(args):
    """`RunRequest` із розібраних аргументів.

    Окремо від `main()`, щоб перевірити, що прапорці доходять до запиту
    без спотворень: `--sequential` і `--time_limit` дають дві похідні
    величини (`effective_cores`, `solver_time_limit`), і саме там колись
    губився режим відтворюваності.
    """
    return RunRequest(
        inp_file=os.path.abspath(args.inp), cost_file=os.path.abspath(args.costs),
        h_min=args.hmin, units=args.units, v_opt=args.v_opt, runs=args.runs,
        cores=args.cores, max_sims=args.max_sims, run_mode=args.run_mode,
        time_limit=args.time_limit, sequential=args.sequential,
    )


def main():
    multiprocessing.freeze_support()

    args, _unknown = build_parser().parse_known_args()
    apply_config_file(args)

    # Порядок обходу множин з рядковими ключами (імена вузлів мережі) залежить
    # від PYTHONHASHSEED, який Python рандомізує на кожен процес. Через це
    # прогін з тим самим --seed давав різні результати. Змінну неможливо
    # застосувати після старту інтерпретатора, тому перезапускаємо процес.
    # Замір на Hanoi: без цього 4 прогони дали 3 різні результати, з ним —
    # 4 однакові (6.1883 M$, 40 905 симуляцій).
    if args.seed is not None and os.environ.get("PYTHONHASHSEED") != "0":
        if getattr(sys, "frozen", False):
            print("[WARNING] Збірка PyInstaller: PYTHONHASHSEED не фіксується, "
                  "прогін не буде побітово відтворюваним.")
        else:
            os.environ["PYTHONHASHSEED"] = "0"
            os.execv(sys.executable, [sys.executable] + sys.argv)

    # Сіється RNG головного процесу. Від нього залежать:
    #  * `solve_fast` цілком (свіп швидкостей + gradient_squeeze виконуються тут);
    #  * `seed_modifier` в `solve_standalone`, з якого воркери дістають власні seed.
    # Без цього жоден прогін не відтворюваний — і `--seed` тут єдина точка входу.
    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)

    # Корінь закріплюється за цим прогоном: `claim_temp_root` створює
    # `_temp_sim_data/run_<pid>` і передає його воркерам. Раніше тут
    # стояло `clean_temp_root()`, яке зносило **спільний** корінь — тобто
    # прогін, стартуючи, вбивав сусідній, що вже працював.
    claim_temp_root()

    base_dir = setup_run_directory()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(base_dir, "logs", f"run_{timestamp}.txt")
    sys.stdout = DualLogger(log_file)
    sys.stderr = sys.stdout

    request = request_from_args(args)

    print("==============================================")
    print(f"   ANALYTICAL OPTIMIZER: {os.path.basename(request.inp_file)}")
    print(f"   Mode: {request.run_mode.upper()} | Cores: {request.effective_cores}")
    print(f"   Seed: {args.seed if args.seed is not None else 'random (not reproducible)'}")
    print("==============================================")

    runner = OptimizationRunner(request)
    try:
        runner.prepare()
    except Exception as e:
        print(f"[CRITICAL] PRE-FLIGHT CHECK FAILED: {e}")
        return

    try:
        results = runner.run(base_dir)
        runner.export(results, base_dir)
    finally:
        runner.close()
        clean_temp_root()


if __name__ == "__main__":
    main()
