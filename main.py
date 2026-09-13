import argparse
import os
import shutil
import time
import pandas as pd
import multiprocessing
import signal
import sys
import traceback
import json

from ga_config import GAConfig
from ga_utils import silence_warnings, format_time
from ga_data import load_config
from water_sim import WaterSimulator
from analytical_solver import AnalyticalSolver
from plot import plot_convergence, plot_network_map, export_solution
from ga_utils import DualLogger
from datetime import datetime

silence_warnings()

worker_sim_instance = None
worker_temp_dir = None

def log_err(msg):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

def get_temp_root():
    return os.path.abspath("_temp_sim_data")

def worker_init(inp_file, config_obj):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    global worker_sim_instance, worker_temp_dir
    
    try:
        pid = os.getpid()
        root = get_temp_root()
        worker_temp_dir = os.path.join(root, f"worker_{pid}")
        os.makedirs(worker_temp_dir, exist_ok=True)

        os.chdir(worker_temp_dir)

        import shutil
        base_name = os.path.splitext(os.path.basename(inp_file))[0]
        local_inp = f"{base_name}_worker_{pid}.inp"
        local_inp = os.path.abspath(local_inp)
        
        shutil.copy2(inp_file, local_inp)
        
        worker_sim_instance = WaterSimulator(local_inp, config_obj, temp_dir=worker_temp_dir)
        
    except Exception as e:
        log_err(f"!!! WORKER {os.getpid()} INIT CRASH: {e}")
        sys.exit(1)

def clean_all_temp():
    root = get_temp_root()
    if os.path.exists(root):
        try:
            shutil.rmtree(root, ignore_errors=True)
        except: 
            time.sleep(0.5)
            try: shutil.rmtree(root, ignore_errors=True)
            except: pass

def setup_run_directory(base_dir="OutputDataExperiments"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    run_dir = os.path.join(base_dir, timestamp)
    
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"[System] Created new output directory: {run_dir}")
    return run_dir

def analytical_worker_task(args):
    diams            = args[0]
    v_opt            = args[1]
    time_budget      = args[2]
    global_best_cost = args[3]
    global_archive   = args[4]
    seed_mod         = args[5]
    worker_id        = args[6]
    shared_progress  = args[7]
    log_dir          = args[8]
    epoch            = args[9]
    max_sims         = args[10] if len(args) > 10 else float('inf')
    n_workers        = args[11] if len(args) > 11 else (len(shared_progress) if shared_progress else 1)
    feasibility_kw   = args[12] if len(args) > 12 else {}

    global worker_sim_instance
    if worker_sim_instance is None:
        raise RuntimeError("Worker simulator not initialized!")

    import random
    import numpy as np
    import os
    import sys
    
    random.seed(seed_mod)
    np.random.seed(seed_mod)

    original_stdout = sys.stdout
    log_file_path = None
    log_file_handle = None

    if log_dir:
        logs_folder = os.path.join(log_dir, "logs") if "logs" not in log_dir else log_dir
        os.makedirs(logs_folder, exist_ok=True)
        log_file_path = os.path.join(logs_folder, f"worker_{worker_id+1:02d}.txt")
        
        log_file_handle = open(log_file_path, "a", encoding="utf-8")
        log_file_handle.write(f"\n\n{'='*50}\n 🚀 STARTING EPOCH {epoch+1} | WORKER {worker_id+1:02d}\n{'='*50}\n")
        log_file_handle.flush()
        
        class WorkerLogger:
            def __init__(self, file_handle):
                self.file_handle = file_handle
            def write(self, message):
                self.file_handle.write(message)
                self.file_handle.flush()
            def writelines(self, lines):
                self.file_handle.writelines(lines)
                self.file_handle.flush()
            def flush(self):
                self.file_handle.flush()

    try:
        if log_file_handle:
            sys.stdout = WorkerLogger(log_file_handle)
        
        from analytical_solver import AnalyticalSolver
        solver_module = sys.modules[AnalyticalSolver.__module__]
        
        SolverContext = solver_module.SolverContext
        LocalSearch = solver_module.LocalSearch
        KickStrategies = solver_module.KickStrategies
        IslandWorker = solver_module.IslandWorker
        FeasibilityPolicy = solver_module.FeasibilityPolicy

        ctx = SolverContext(worker_sim_instance, diams, v_opt=v_opt)
        ctx.log_file = log_file_path 
        
        ls = LocalSearch(ctx)
        kicker = KickStrategies(ctx, ls)
        
        n = ctx.num_pipes
        network_class = "SMALL" if n < 50 else ("MEDIUM" if n < 200 else ("LARGE" if n < 1000 else "XLARGE"))
        beam_width = 8 if network_class in ["LARGE", "XLARGE"] else 5
        
        worker = IslandWorker(ctx, kicker, ls, worker_id, n_workers, max_sims, beam_width, network_class, global_archive, epoch,
                              feasibility=FeasibilityPolicy(**feasibility_kw))
        c_best, sol_best = worker.run(time_budget, global_best_cost, shared_progress)
        
        return c_best, sol_best, None, ctx.sim_count, worker.pool.basin_tabu
        
    finally:
        sys.stdout = original_stdout
        if log_file_handle:
            log_file_handle.close()

def main():
    multiprocessing.freeze_support()

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
    parser.add_argument("--dive", type=str, choices=["off", "relax", "dive"], default="off",
                        help="SUBMARINE — стратегічна осциляція навколо межі допустимості. "
                             "off: без додаткової поступки (поведінка за замовчуванням). "
                             "relax: стала поступка --dive_depth без фази випливання. "
                             "dive: повний цикл із примусовим випливанням і відкатом.")
    parser.add_argument("--dive_depth", type=float, default=0.5,
                        help="Глибина занурення в метрах нижче hmin (для --dive relax|dive).")
    parser.add_argument("--dive_rounds", type=int, default=8,
                        help="Скільки раундів триває одне занурення (для --dive dive).")
    parser.add_argument("--dive_surface", type=str, choices=["cheapest", "farthest"], default="cheapest",
                        help="Звідки випливати: cheapest — найдешевше рішення пулу; "
                             "farthest — найвіддаленіше за Хеммінгом від точки занурення.")
    parser.add_argument("--dive_cooldown", type=int, default=12,
                        help="Пауза в раундах між зануреннями.")
    parser.add_argument("--dive_attempts", type=int, default=1,
                        help="Скільки кандидатів пробувати підняти за одне випливання.")
    parser.add_argument("--time_limit", type=float, default=None,
                        help="Wall-clock limit per run in seconds; 0 = unlimited (stop only on "
                             "--max_sims). Default: derived from the calibrated simulator speed, "
                             "which makes the stopping point machine- and load-dependent.")

    args, unknown = parser.parse_known_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            cli_args_used = [arg.strip('-') for arg in sys.argv if arg.startswith('--')]
            for key, value in config_data.items():
                if key not in cli_args_used and hasattr(args, key):
                    setattr(args, key, value)

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

    args.inp = os.path.abspath(args.inp)
    args.costs = os.path.abspath(args.costs)

    # Сіється RNG головного процесу. Від нього залежать:
    #  * `solve_fast` цілком (свіп швидкостей + gradient_squeeze виконуються тут);
    #  * `seed_modifier` в `solve_standalone`, з якого воркери дістають власні seed.
    # Без цього жоден прогін не відтворюваний — і `--seed` тут єдина точка входу.
    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)

    clean_all_temp()
    os.makedirs(get_temp_root(), exist_ok=True)
    main_proc_temp = os.path.join(get_temp_root(), "main_process")
    os.makedirs(main_proc_temp, exist_ok=True)
    
    base_dir = setup_run_directory()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{base_dir}/logs/run_{timestamp}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    sys.stdout = DualLogger(log_file)
    sys.stderr = sys.stdout
    
    config = GAConfig(
        cost_file=args.costs,
        h_min=args.hmin,
        unit_system=args.units,
        run_mode=args.run_mode,
        v_opt=args.v_opt
    )
    load_config(config)

    try:
        sim = WaterSimulator(args.inp, config, temp_dir=main_proc_temp)
        sim.evaluate([0] * sim.n_variables, penalty_factor=1000, epsilon=0.0)
    except Exception as e:
        print(f"[CRITICAL] PRE-FLIGHT CHECK FAILED: {e}")
        return

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "tables"), exist_ok=True)
    
    num_cores = args.cores if args.cores > 0 else multiprocessing.cpu_count()
    print("==============================================")
    print(f"   ANALYTICAL OPTIMIZER: {os.path.basename(args.inp)}")
    print(f"   Mode: {args.run_mode.upper()} | Cores: {num_cores}")
    print(f"   Seed: {args.seed if args.seed is not None else 'random (not reproducible)'}")
    if args.dive != "off":
        print(f"   SUBMARINE: {args.dive} | depth={args.dive_depth} м | rounds={args.dive_rounds} "
              f"| surface={args.dive_surface} x{args.dive_attempts}")
    print("==============================================")
    
    pool = None
    results = []
    try:
        global worker_sim_instance
        if args.sequential:
            # Один воркер у цьому ж процесі, без Pool і без Manager. Тільки так
            # прогін справді відтворюваний: у паралельному режимі глобальний
            # архів наповнює монітор із таймером, тому гілки, що залежать від
            # розміру архіву (BASIN_ESCAPE, consensus locking), спрацьовують
            # у різні моменти навіть при однаковому seed.
            num_cores = 1
            worker_sim_instance = sim
            print("   [Sequential] Pool вимкнено — прогін детермінований\n")
            if args.time_limit is None:
                # Часовий бюджет за замовчуванням виводиться з калібрування
                # швидкості симулятора, тому точка зупинки залежить від
                # завантаження машини. Для детермінованого прогону зупиняємось
                # виключно за лічильником симуляцій.
                args.time_limit = 0
        else:
            pool = multiprocessing.Pool(processes=num_cores, initializer=worker_init, initargs=(args.inp, config))
        AnalyticalSolver.worker_task = staticmethod(analytical_worker_task)

        for i in range(args.runs):
            current_log_dir = os.path.abspath(base_dir) if args.runs == 1 else os.path.abspath(os.path.join(base_dir, f"run_{i+1}"))
            if args.runs > 1: os.makedirs(current_log_dir, exist_ok=True)

            time_limit = None if args.time_limit is None else (
                float('inf') if args.time_limit <= 0 else args.time_limit
            )
            feasibility_kwargs = {
                "mode": args.dive,
                "depth": args.dive_depth,
                "max_rounds": args.dive_rounds,
                "surface_from": args.dive_surface,
                "surface_attempts": args.dive_attempts,
                "cooldown_rounds": args.dive_cooldown,
            }
            solver = AnalyticalSolver(
                sim, config.diameters_m, v_opt=config.v_opt, pool=pool,
                log_dir=current_log_dir, n_workers=num_cores, max_sims=args.max_sims,
                time_limit_sec=time_limit, feasibility_kwargs=feasibility_kwargs
            )
            
            start_t = time.time()
            if config.run_mode == 'fast_analytical':
                best_solution_meters = solver.solve_fast()
            else:
                best_solution_meters = solver.solve_standalone()
            duration = time.time() - start_t
        
            best_indices = []
            for d in best_solution_meters:
                try: idx = config.diameters_m.index(d)
                except ValueError: idx = len(config.diameters_m) - 1
                best_indices.append(idx)
            
            final_cost, final_p, _, _ = sim.get_stats(best_indices)
            is_feasible = (final_p >= args.hmin)
            status = "OK" if is_feasible else "FAIL"
        
            print("\n=== RESULTS ===")
            print(f"Run {i+1}: {final_cost/1e6:.4f}M$ | P={final_p:.2f}m | {status} | Time: {format_time(duration)}")
            
            if args.runs > 1:
                local_tables_dir = os.path.join(current_log_dir, "tables")
                os.makedirs(local_tables_dir, exist_ok=True)
                export_solution(best_indices, [], args.inp, os.path.join(local_tables_dir, "solution"), config)
            
            raw_hist = getattr(solver, 'history', [])
            if not raw_hist: raw_hist = [(solver.ctx.sim_count, final_cost)]
            formatted_hist = [{"evals": sc, "min_cost": bc} for sc, bc in raw_hist]
            
            results.append({
                "run_id": i+1, "cost": final_cost, "pressure": final_p,
                "feasible": is_feasible, "time": duration, "individual": best_indices, "history": formatted_hist
            })

    except KeyboardInterrupt:
        print("\n[System] 🛑 User interrupted via Keyboard (Ctrl+C). Terminating...")
    except Exception as e:
        print(f"\n[Fatal Error] {e}"); traceback.print_exc()
    finally:
        if pool: pool.terminate(); pool.join()
        try: time.sleep(1.0); clean_all_temp()
        except: pass

    if not results: return

    best_run = sorted(results, key=lambda x: (not x['feasible'], x['cost']))[0]
    
    print("\n=== FINAL RESULTS ===")
    for res in results:
        status = "OK" if res['feasible'] else "FAIL"
        print(f"Run {res['run_id']}: {res['cost']/1e6:.4f}M$ | P={res['pressure']:.2f}m | {status}")

    total_evals = best_run['history'][-1].get('evals', len(best_run['history'])) if best_run['history'] else 0

    solution_path = os.path.join(base_dir, "tables", "solution_champion")
    export_solution(
        individual=best_run['individual'], history=best_run['history'], inp_file=args.inp, 
        filename_prefix=solution_path, config=config, cost=best_run['cost'], time_sec=best_run['time'], total_sims=total_evals          
    )
    
    plot_network_map(best_run['individual'], args.inp, os.path.join(base_dir, "plots", "network_map.png"), config, best_run['cost'])
    
    if len(best_run['history']) > 1:
        plot_convergence(best_run['history'], os.path.join(base_dir, "plots", "convergence.png"))
    
    summary_data = [{"Run": r['run_id'], "Cost": r['cost'], "Pressure": r['pressure'], "Feasible": r['feasible'], "Time": r['time']} for r in results]
    pd.DataFrame(summary_data).to_csv(os.path.join(base_dir, "tables", "runs_summary.csv"), index=False)

if __name__ == "__main__":
    main()