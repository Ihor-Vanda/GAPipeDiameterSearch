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
from ga_config import *
from ga_utils import silence_warnings, format_time
from ga_data import load_config
from water_sim import WaterSimulator
from ga_optimizer import GeneticOptimizer
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

def worker_init(inp_file, config_data):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    global worker_sim_instance, worker_temp_dir
    
    try:
        pid = os.getpid()
        root = get_temp_root()
        worker_temp_dir = os.path.join(root, f"worker_{pid}")
        os.makedirs(worker_temp_dir, exist_ok=True)

        os.chdir(worker_temp_dir)

        CONFIG.clear()
        CONFIG.update(config_data)
        
        if 'diameters_raw' not in CONFIG or len(CONFIG['diameters_raw']) == 0:
             raise ValueError("Worker received empty config!")

        worker_sim_instance = WaterSimulator(inp_file)
        
    except Exception as e:
        log_err(f"!!! WORKER {os.getpid()} INIT CRASH: {e}")
        sys.exit(1)

def worker_eval_task(args):
    ind, gen, pf, epsilon = args
    global worker_sim_instance
    
    if worker_sim_instance is None:
        return (float('inf'),)

    unique_name = f"sim_{uuid.uuid4().hex[:8]}"
    
    try:
        return worker_sim_instance.evaluate(ind, penalty_factor=pf, epsilon=epsilon, file_prefix=unique_name)
    except Exception:
        return (float('inf'),)
    finally:
        try:
            for f in os.listdir('.'):
                if f.startswith(unique_name) or f.lower() in ["temp.inp", "temp.rpt", "temp.bin"]:
                    try: os.remove(f)
                    except: pass
        except: pass

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

def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Hybrid Genetic Algorithm for WDN Optimization")
    parser.add_argument("--inp", type=str, default=DEFAULT_INP_FILE, help="Path to EPANET .inp file")
    parser.add_argument("--costs", type=str, default=DEFAULT_COST_FILE, help="Path to cost configuration JSON")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of independent runs")
    parser.add_argument("--pop", type=int, default=0, help="Population size (0 = auto)") 
    parser.add_argument("--gen", type=int, default=0, help="Max generations (0 = auto)")
    parser.add_argument("--hmin", type=float, default=DEFAULT_H_MIN, help="Minimum allowable head (pressure)")
    parser.add_argument("--units", type=str, choices=["in", "mm"], default="mm", help="Units for pipe diameters")
    parser.add_argument("--init", type=str, choices=["random", "static", "hyper"], default="hyper", help="Initialization strategy")
    parser.add_argument("--cores", type=int, default=0, help="Number of CPU cores (0 = all)")
    parser.add_argument("--fixed", action="store_true", help="Enable fixed penalty mode")
    
    parser.add_argument("--no-eps", action="store_true", help="Disable adaptive epsilon relaxation")
    parser.add_argument("--no-shocks", action="store_true", help="Disable seismic shocks (re-starts)")
    parser.add_argument("--no-expansion", action="store_true", help="Disable population expansion on stagnation")
    parser.add_argument("--no-graphs", action="store_true", help="Disable graph heuristics (Smart Repair & Squeeze)")

    args = parser.parse_args()

    args.inp = os.path.abspath(args.inp)
    args.costs = os.path.abspath(args.costs)

    clean_all_temp()
    os.makedirs(get_temp_root(), exist_ok=True)

    main_proc_temp = os.path.join(get_temp_root(), "main_process")
    os.makedirs(main_proc_temp, exist_ok=True)
    
    base_dir = setup_run_directory()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{base_dir}/logs/run_{timestamp}.txt"
    
    sys.stdout = DualLogger(log_file)
    sys.stderr = sys.stdout
    
    print(f"[System] Log file initiated: {log_file}")

    print("[System] Loading configuration in Main Process...")
    try:
        load_config(args.costs, args.hmin, args.units)
        config_snapshot = copy.deepcopy(CONFIG)
    except Exception as e:
        print(f"[Main Error] Config loading failed: {e}")
        return

    try:
        print("[System] Pre-flight simulation check...")
        sim = WaterSimulator(args.inp, temp_dir=main_proc_temp)
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
            initargs=(args.inp, config_snapshot)
        )

        WaterSimulator.worker_eval_wrapper = staticmethod(worker_eval_task)

        results = []
        optimizer = GeneticOptimizer(sim, args.pop, args.gen, pool=pool, fixed_mode=args.fixed)
        
        use_epsilon = not args.no_eps
        use_shocks = not args.no_shocks
        use_expansion = not args.no_expansion
        use_graphs = not args.no_graphs

        for i in range(args.runs):
            optimizer.total_sims = 0
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
    export_solution(best_run['individual'], best_run['history'], args.inp, solution_path)
    plot_convergence(best_run['history'], os.path.join(base_dir, "plots", "convergence.png"))
    plot_network_map(best_run['individual'], args.inp, os.path.join(base_dir, "plots", "network_map.png"))
    
    summary_data = [{"Run": r['run_id'], "Cost": r['cost'], "Pressure": r['pressure'], "Feasible": r['feasible'], "Time": r['time']} for r in results]
    pd.DataFrame(summary_data).to_csv(os.path.join(base_dir, "tables", "runs_summary.csv"), index=False)

if __name__ == "__main__":
    main()