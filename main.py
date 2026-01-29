import argparse
import os
import time
import pandas as pd
import wntr
from ga_config import *
from ga_utils import silence_warnings, format_time
from ga_data import load_config
from ga_sim import NetworkSimulator
from ga_optimizer import GeneticOptimizer
from ga_plot import plot_convergence, plot_network_map, export_solution

silence_warnings()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default=DEFAULT_INP_FILE)
    parser.add_argument("--costs", type=str, default=DEFAULT_COST_FILE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--pop", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--gen", type=int, default=DEFAULT_GENS)
    parser.add_argument("--hmin", type=float, default=DEFAULT_H_MIN)
    parser.add_argument("--units", type=str, choices=["in", "mm"], default="mm")
    args = parser.parse_args()

    load_config(args.costs, args.hmin, args.units)

    print("==============================================")
    print(f"   EVOLUTIONARY OPTIMIZER: {args.inp}")
    print("==============================================")
    simulator = NetworkSimulator(args.inp)

    results = []
    optimizer = GeneticOptimizer(simulator, args.pop, args.gen)

    for i in range(args.runs):
        ind, cost, p, dur, hist = optimizer.run(i, args.hmin)
        
        # --- POST-PROCESS ---
        print("    [Post-Process] Refining Solution...")
        
        if simulator.n_pipes < 100:
            polished_ind = optimizer.run_kick_and_fix(ind, iterations=100, kick_strength=3, verbose=True)
        else:
            limit = 500
            polished_ind = optimizer.run_local_search(ind, limit_pipes=limit, verbose=True)
        
        final_cost, final_p = simulator.get_stats(polished_ind)
        is_feasible = (final_p >= args.hmin - 0.001)
        
        results.append({
            "run_id": i+1, "cost": final_cost, "pressure": final_p,
            "feasible": is_feasible, "time": dur, "individual": polished_ind, "history": hist
        })
        
    best_run = sorted(results, key=lambda x: (not x['feasible'], x['cost']))[0]
    
    print("\n=== FINAL RESULTS ===")
    for res in results:
        status = "OK" if res['feasible'] else "FAIL"
        print(f"Run {res['run_id']}: {res['cost']/1e6:.4f}M$ | P={res['pressure']:.2f}m | {status}")

    wn = wntr.network.WaterNetworkModel(args.inp)
    pipe_data = []
    for i, pipe_name in enumerate(wn.pipe_name_list):
        idx = best_run['individual'][i]
        link = wn.get_link(pipe_name)
        unit = "mm" if CONFIG["unit_system"] == "mm" else "in"
        pipe_data.append({
            "Pipe": pipe_name, 
            f"Diam ({unit})": CONFIG["diameters_raw"][idx],
            "Cost": link.length * CONFIG["costs"][CONFIG["diameters_raw"][idx]]
        })
    pd.DataFrame(pipe_data).to_csv("solution_champion.csv", index=False)
    
    plot_convergence(best_run['history'], "solution_champion_convergence.png")
    plot_network_map(best_run['individual'], args.inp, "solution_champion_map.png")

if __name__ == "__main__":
    main()