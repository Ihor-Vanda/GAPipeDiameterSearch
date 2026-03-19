import argparse
import os
import shutil
import time
import pandas as pd
import multiprocessing
import uuid
import signal
import sys
import traceback
import copy
import json

from ga_config import GAConfig
from ga_utils import silence_warnings, format_time
from ga_data import load_config
from water_sim import WaterSimulator
from ga_optimizer import GeneticOptimizer
from analytical_solver import AnalyticalSolver
from ga_plot import plot_convergence, plot_network_map, export_solution
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

def worker_eval_task(args):
    ind, gen, pf, epsilon = args
    global worker_sim_instance
    
    if worker_sim_instance is None:
        return (float('inf'),)

    try:
        val = worker_sim_instance.evaluate(ind, penalty_factor=pf, epsilon=epsilon)
        
        if not isinstance(val, tuple): 
            val = (val,)
        return val
        
    except Exception as e:
        log_err(f"Worker Eval Error: {e}")
        return (float('inf'),)

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
    # 1. Безпечне розпакування за індексами
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
    failed_basins    = args[10]
    max_sims         = args[11] if len(args) > 11 else float('inf')
    n_workers        = args[12] if len(args) > 12 else (len(shared_progress) if shared_progress else 1)

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

        ctx = SolverContext(worker_sim_instance, diams, v_opt=v_opt)
        ctx.log_file = log_file_path 
        
        ls = LocalSearch(ctx)
        kicker = KickStrategies(ctx, ls)
        
        n = ctx.num_pipes
        network_class = "SMALL" if n < 50 else ("MEDIUM" if n < 200 else ("LARGE" if n < 1000 else "XLARGE"))
        beam_width = 8 if network_class in ["LARGE", "XLARGE"] else 5
        
        worker = IslandWorker(ctx, kicker, ls, worker_id, n_workers, max_sims, beam_width, network_class, global_archive, epoch)
        c_best, sol_best = worker.run(time_budget, global_best_cost, shared_progress)
        
        return c_best, sol_best, None, ctx.sim_count, worker.pool.basin_tabu
        
    finally:
        sys.stdout = original_stdout
        if log_file_handle:
            log_file_handle.close()

def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Hybrid Genetic Algorithm / Analytical Solver for WDN")
    
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (overrides other args)")
    parser.add_argument("--max_sims", type=int, default=None, help="Global simulation budget (e.g., 15000000)")
    
    parser.add_argument("--inp", type=str, default="InputData/Hanoi/Hanoi.inp", help="Path to EPANET .inp file")
    parser.add_argument("--costs", type=str, default="InputData/Hanoi/costs.csv", help="Path to cost configuration JSON")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--pop", type=int, default=0, help="Population size (0 = auto)") 
    parser.add_argument("--gen", type=int, default=0, help="Max generations (0 = auto)")
    parser.add_argument("--hmin", type=float, default=30.0, help="Minimum allowable head (pressure)")
    parser.add_argument("--units", type=str, choices=["in", "mm"], default="mm", help="Units for pipe diameters")
    parser.add_argument("--cores", type=int, default=0, help="Number of CPU cores (0 = all)")
    parser.add_argument("--fixed", action="store_true", help="Enable fixed penalty mode")
    
    parser.add_argument("--run_mode", type=str, choices=["ga", "analytical"], default="ga", help="Mode: ga (Genetic Algorithm) or analytical")
    parser.add_argument("--init", type=str, choices=["random", "static", "sep", "analytical"], default="sep", help="Initialization strategy for GA")
    parser.add_argument("--v_opt", type=float, default=1.0, help="Optimal velocity (m/s) for analytical solver")

    parser.add_argument("--no-eps", action="store_true", help="Disable adaptive epsilon relaxation")
    parser.add_argument("--no-shocks", action="store_true", help="Disable seismic shocks (re-starts)")
    parser.add_argument("--no-expansion", action="store_true", help="Disable population expansion on stagnation")
    parser.add_argument("--no-graphs", action="store_true", help="Disable graph heuristics (Smart Repair & Squeeze)")

    args = parser.parse_args()

    if args.config:
        if os.path.exists(args.config):
            print(f"[System] Loading configuration from {args.config}...")
            with open(args.config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    setattr(args, key, value)
        else:
            print(f"[Fatal Error] Config file not found: {args.config}")
            sys.exit(1)
    args.inp = os.path.abspath(args.inp)
    args.costs = os.path.abspath(args.costs)

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
    
    print(f"[System] Log file initiated: {log_file}")

    print("[System] Loading configuration in Main Process...")
    
    config = GAConfig(
        inp_file=args.inp,
        cost_file=args.costs,
        pop_size=args.pop if args.pop > 0 else 200,
        n_gens=args.gen if args.gen > 0 else 150,
        runs=args.runs,
        h_min=args.hmin,
        unit_system=args.units,
        run_mode=args.run_mode,
        init_method=args.init,
        v_opt=args.v_opt
    )
    
    try:
        load_config(config)
    except Exception as e:
        print(f"[Main Error] Config loading failed: {e}")
        return

    try:
        print("[System] Pre-flight simulation check...")
        sim = WaterSimulator(args.inp, config, temp_dir=main_proc_temp)
        test_ind = [0] * sim.n_variables
        sim.evaluate(test_ind, penalty_factor=1000, epsilon=0.0)
        print("[System] Pre-flight check PASSED ✅")
    except Exception as e:
        print(f"[CRITICAL] PRE-FLIGHT CHECK FAILED: {e}")
        traceback.print_exc()
        return

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "tables"), exist_ok=True)
    
    print("==============================================")
    print(f"   EVOLUTIONARY OPTIMIZER: {os.path.basename(args.inp)}")
    print(f"   System: {os.name.upper()} | Cores: {args.cores if args.cores > 0 else 'Auto'}")
    print("==============================================")
    
    num_cores = args.cores if args.cores > 0 else multiprocessing.cpu_count()
    print(f"[System] Initializing {num_cores} parallel workers (Strict Isolation)...")
    
    pool = None
    try:
        pool = multiprocessing.Pool(
            processes=num_cores, 
            initializer=worker_init, 
            initargs=(args.inp, config)
        )

        WaterSimulator.worker_eval_wrapper = staticmethod(worker_eval_task)
        AnalyticalSolver.worker_task = staticmethod(analytical_worker_task)
        
        results = []
        
        if config.run_mode == 'analytical':
            print(f"\n[Mode] Running Fast Analytical Solver (v_opt = {config.v_opt} m/s)...")
            
            abs_log_dir = os.path.abspath(base_dir)
            solver = AnalyticalSolver(
                sim, 
                config.diameters_m, 
                v_opt=config.v_opt, 
                pool=pool, 
                log_dir=abs_log_dir, 
                n_workers=num_cores, 
                max_sims=args.max_sims
            )
            
            start_t = time.time()
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
            
            print("\n=== FINAL RESULTS ===")
            print(f"Analytical Run: {final_cost/1e6:.4f}M$ | P={final_p:.2f}m | {status} | Time: {format_time(duration)}")
            export_solution(best_indices, [], args.inp, os.path.join(base_dir, "tables", "analytical_solution"), config)
            
        else:
            optimizer = GeneticOptimizer(sim, config, pop_size=args.pop, n_gens=args.gen, pool=pool, fixed_mode=args.fixed)
            
            use_epsilon = not args.no_eps
            use_shocks = not args.no_shocks
            use_expansion = not args.no_expansion
            use_graphs = not args.no_graphs

            for i in range(args.runs):
                optimizer.total_sims = 0
                ind, _, _, dur, hist = optimizer.run(
                    run_id=i, 
                    h_min=args.hmin, 
                    init_mode=args.init, 
                    use_epsilon=use_epsilon,
                    use_shocks=use_shocks,
                    use_expansion=use_expansion,
                    use_graph_heuristics=use_graphs
                )
                
                print("    [Post-Process] Refining Solution...")
                if use_graphs:
                    polished_ind = optimizer.run_local_search(ind, limit_pipes=500)
                else:
                    polished_ind = list(ind)
                
                final_cost, final_p, _, _ = sim.get_stats(polished_ind)
                is_feasible = (final_p >= args.hmin + 0.000001)
                
                results.append({
                    "run_id": i+1, "cost": final_cost, "pressure": final_p,
                    "feasible": is_feasible, "time": dur, "individual": polished_ind, "history": hist
                })

                pd.DataFrame(hist).to_csv(os.path.join(base_dir, "tables", f"run_{i+1}_history.csv"), index=False)
            
    except KeyboardInterrupt:
        print("\n[System] 🛑 User interrupted via Keyboard (Ctrl+C). Terminating...")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        traceback.print_exc()
    finally:
        if pool:
            print("[System] Shutting down workers pool...")
            pool.terminate()
            pool.join()
        
        try:
            time.sleep(1.0)
            clean_all_temp()
        except: pass
        print("[System] Cleanup complete.")
        
    if not results: return

    best_run = sorted(results, key=lambda x: (not x['feasible'], x['cost']))[0]
    
    print("\n=== FINAL RESULTS ===")
    for res in results:
        status = "OK" if res['feasible'] else "FAIL"
        print(f"Run {res['run_id']}: {res['cost']/1e6:.4f}M$ | P={res['pressure']:.2f}m | {status}")

    solution_path = os.path.join(base_dir, "tables", "solution_champion")
    export_solution(best_run['individual'], best_run['history'], args.inp, solution_path, config)
    plot_network_map(best_run['individual'], args.inp, os.path.join(base_dir, "plots", "network_map.png"), config)
    
    plot_convergence(best_run['history'], os.path.join(base_dir, "plots", "convergence.png"))
    summary_data = [{"Run": r['run_id'], "Cost": r['cost'], "Pressure": r['pressure'], "Feasible": r['feasible'], "Time": r['time']} for r in results]
    pd.DataFrame(summary_data).to_csv(os.path.join(base_dir, "tables", "runs_summary.csv"), index=False)

if __name__ == "__main__":
    main()