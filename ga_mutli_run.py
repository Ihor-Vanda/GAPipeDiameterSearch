# ==========================================
#        SILENCE WARNINGS (MUST BE FIRST)
# ==========================================
import warnings
import logging

warnings.simplefilter("ignore")
logging.getLogger("wntr").setLevel(logging.ERROR)

# ==========================================
#           IMPORTS
# ==========================================
import wntr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from deap import base, creator, tools
import random
import time
import csv
import argparse
import os
import sys

# ==========================================
#        DEFAULT CONFIGURATION
# ==========================================
DEFAULT_INP_FILE = "Balerma_Clean.inp"
DEFAULT_COST_FILE = "costs_balerma.csv"
DEFAULT_POPSIZE = 100
DEFAULT_GENS = 100
DEFAULT_RUNS = 1
DEFAULT_H_MIN = 20.0

MUTATION_START = 0.40   
MUTATION_END = 0.10     
EPSILON_START = 10.0    
EPSILON_END = 0.0       

CONFIG = {
    "diameters_raw": [],
    "diameters_m": [],
    "costs": {},
    "h_min": 30.0,
    "unit_system": "mm" 
}

# --- HELPER: Time Formatter ---
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m} m {s} s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h} h {m} m {s} s"

# ==========================================
#           DATA MANAGEMENT
# ==========================================

def load_config(cost_file, h_min, units):
    if not os.path.exists(cost_file):
        print(f"[Warning] Cost file '{cost_file}' not found.")
    
    try:
        df = pd.read_csv(cost_file)
        df = df.sort_values(by="Diameter")
        
        CONFIG["diameters_raw"] = df["Diameter"].tolist()
        
        if units == "mm":
            CONFIG["diameters_m"] = [d / 1000.0 for d in df["Diameter"]]
            print("[Config] Units: Millimeters (converted to meters / 1000)")
        else:
            CONFIG["diameters_m"] = [d * 0.0254 for d in df["Diameter"]]
            print("[Config] Units: Inches (converted to meters * 0.0254)")
            
        CONFIG["costs"] = dict(zip(df["Diameter"], df["Cost"]))
        CONFIG["h_min"] = h_min
        CONFIG["unit_system"] = units
        
        print(f"[Config] Loaded {len(CONFIG['diameters_raw'])} pipe types.")
        
    except Exception as e:
        print(f"[Error] Failed to load config: {e}")
        sys.exit(1)

# ==========================================
#           CORE FUNCTIONS
# ==========================================

def evaluate_network(individual, inp_file, strategy="static", gen=0, tolerance=0.0, total_gens=100):
    # --- SILENCE BLOCK (Локальне глушіння при завантаженні) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn = wntr.network.WaterNetworkModel(inp_file)
        
    pipe_names = wn.pipe_name_list
    total_cost = 0.0
    
    costs = CONFIG["costs"]
    diams_raw = CONFIG["diameters_raw"]
    diams_m = CONFIG["diameters_m"]
    h_min = CONFIG["h_min"]

    for i, pipe_name in enumerate(pipe_names):
        idx = individual[i]
        total_cost += wn.get_link(pipe_name).length * costs[diams_raw[idx]]
        wn.get_link(pipe_name).diameter = diams_m[idx]

    sim = wntr.sim.EpanetSimulator(wn)
    try:
        results = sim.run_sim()
    except Exception:
        return 1e15, 
    
    pressures = results.node['pressure'].iloc[-1]
    junctions = wn.junction_name_list
    
    violation = 0.0
    effective_limit = h_min - tolerance
    
    for node in junctions:
        p = pressures[node]
        if p < effective_limit:
            violation += (effective_limit - p)

    penalty = 0.0
    if violation > 0:
        penalty = 1e9 + (1e6 * violation)

    return total_cost + penalty,

def get_real_stats(individual, inp_file):
    # --- SILENCE BLOCK ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn = wntr.network.WaterNetworkModel(inp_file)
        
    total_cost = 0.0
    pipe_names = wn.pipe_name_list
    costs = CONFIG["costs"]
    diams_raw = CONFIG["diameters_raw"]
    diams_m = CONFIG["diameters_m"]
    
    for i, pipe_name in enumerate(pipe_names):
        idx = individual[i]
        total_cost += wn.get_link(pipe_name).length * costs[diams_raw[idx]]
        wn.get_link(pipe_name).diameter = diams_m[idx]
    
    try:
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        min_p = results.node['pressure'].iloc[-1].loc[wn.junction_name_list].min()
    except:
        min_p = -999.0
        
    return total_cost, min_p

def mutCreepInt(individual, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            step = random.randint(1, 3)
            change = step if random.random() < 0.5 else -step
            new_val = individual[i] + change
            if new_val < low: new_val = low
            elif new_val > up: new_val = up
            individual[i] = new_val
    return individual,

def create_mixed_population(n_pipes, n_diams, pop_size):
    pop = []
    # 25% Max diameters
    for _ in range(int(pop_size * 0.25)):
        ind = [n_diams - 1] * n_pipes 
        pop.append(creator.Individual(ind))
    # 25% Median diameters
    mid_idx = n_diams // 2
    for _ in range(int(pop_size * 0.25)):
        ind = [mid_idx] * n_pipes
        pop.append(creator.Individual(ind))
    # 50% Random
    remaining = pop_size - len(pop)
    for _ in range(remaining):
        ind = [random.randint(0, n_diams-1) for _ in range(n_pipes)]
        pop.append(creator.Individual(ind))
    return pop

# --- LOCAL SEARCH ---
def simple_descent(individual, inp_file):
    curr = list(individual)
    max_checks = 100 if len(curr) > 300 else len(curr)
    indices = random.sample(range(len(curr)), k=max_checks)
    
    for i in indices:
        if curr[i] > 0:
            candidate = list(curr)
            candidate[i] -= 1
            curr_score = evaluate_network(curr, inp_file, "static", tolerance=0.0)[0]
            cand_score = evaluate_network(candidate, inp_file, "static", tolerance=0.0)[0]
            if cand_score < curr_score:
                curr[i] -= 1
    return curr

def deep_local_search(individual, inp_file, run_id_label=""):
    print(f"\n   > [Run {run_id_label}] Running Deep Local Search...")
    polish_start = time.time()
    
    current_best = list(individual)
    cost_tuple = evaluate_network(current_best, inp_file, "static", tolerance=0.0)
    current_cost = cost_tuple[0]
    
    n_pipes = len(current_best)
    
    if n_pipes < 300:
        check_indices = list(range(n_pipes))
        print("     [Mode] Full Search")
    else:
        sample_size = min(100, n_pipes)
        check_indices = random.sample(range(n_pipes), k=sample_size)
        print(f"     [Mode] Sampling Search ({sample_size}/{n_pipes} pipes)")

    improved = True
    while improved:
        improved = False
        best_move_candidate = None
        best_move_cost = current_cost
        
        for i in check_indices: 
            if current_best[i] > 0: 
                candidate = list(current_best)
                candidate[i] -= 1 
                
                c_tuple = evaluate_network(candidate, inp_file, "static", tolerance=0.0)
                cost = c_tuple[0]
                
                if cost < 1e13:
                    if cost < best_move_cost:
                        best_move_cost = cost
                        best_move_candidate = candidate

        if best_move_candidate is not None and best_move_cost < current_cost - 0.1:
            current_best = best_move_candidate
            current_cost = best_move_cost
            improved = True
            
    print(f"   > Polished in {format_time(time.time() - polish_start)}. Final: {current_cost/1e6:.4f} M$")
    return current_best

# --- REPORTING ---
def export_solution(individual, history, inp_file, filename_prefix="final_solution"):
    print(f"\n--- Saving results to {filename_prefix}.csv ---")
    
    # --- SILENCE BLOCK ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn = wntr.network.WaterNetworkModel(inp_file)
        
    pipe_data = []
    costs = CONFIG["costs"]
    diams_raw = CONFIG["diameters_raw"]
    
    for i, pipe_name in enumerate(wn.pipe_name_list):
        idx = individual[i]
        link = wn.get_link(pipe_name)
        unit_label = "mm" if CONFIG["unit_system"] == "mm" else "inch"
        
        pipe_data.append({
            "Pipe ID": pipe_name,
            f"Diameter ({unit_label})": diams_raw[idx],
            "Cost": link.length * costs[diams_raw[idx]]
        })
    pd.DataFrame(pipe_data).to_csv(f"{filename_prefix}.csv", index=False)

# ==========================================
#           RUN LOGIC
# ==========================================

def run_single_trial(run_id, args):
    random.seed(time.time() + run_id)
    run_start = time.time()
    print(f"\n>>> Starting Run #{run_id + 1}/{args.runs}...")
    
    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    n_diams = len(CONFIG["diameters_raw"])
    
    # --- SILENCE BLOCK ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn_temp = wntr.network.WaterNetworkModel(args.inp)
        
    n_pipes = len(wn_temp.pipe_name_list)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("attr_int", random.randint, 0, n_diams-1)

    # 1. POPULATION
    pop = create_mixed_population(n_pipes, n_diams, args.pop)
    
    # 2. PRE-FLIGHT CHECK
    if run_id == 0:
        max_ind = [n_diams - 1] * n_pipes
        _, max_p = get_real_stats(max_ind, args.inp)
        print(f"    [INFO] Feasibility Check (Max Pipes): P={max_p:.2f} m")
        if max_p < args.hmin:
            print(f"    [CRITICAL WARNING] Target {args.hmin}m is IMPOSSIBLE. Max feasible is {max_p:.2f} m.")
    
    hof = tools.HallOfFame(1)
    history_log = [] 
    
    fitnesses = [evaluate_network(ind, args.inp, "epsilon", 0, EPSILON_START, args.gen) for ind in pop]
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    hof.update(pop)

    for gen in range(args.gen):
        if gen < args.gen * 0.75: 
            progress = gen / (args.gen * 0.75)
            tol = EPSILON_START - progress * (EPSILON_START - EPSILON_END)
        else: tol = 0.0

        mut_prob = MUTATION_START - (gen/args.gen)*(MUTATION_START - MUTATION_END)
        toolbox.register("mutate", mutCreepInt, low=0, up=n_diams-1, indpb=mut_prob)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8: toolbox.mate(child1, child2); del child1.fitness.values, child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.35: toolbox.mutate(mutant); del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [evaluate_network(ind, args.inp, "epsilon", gen, tol, args.gen) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
        
        pop[:] = offspring
        
        if gen > 0 and gen % 5 == 0 and len(hof) > 0:
            imp_ind_list = simple_descent(list(hof[0]), args.inp)
            imp_ind = creator.Individual(imp_ind_list)
            imp_fit = evaluate_network(imp_ind, args.inp, "epsilon", gen, tol, args.gen)
            imp_ind.fitness.values = imp_fit
            if imp_fit[0] < hof[0].fitness.values[0]:
                hof[0] = imp_ind
                pop[random.randint(0, len(pop)-1)] = imp_ind

        best_cand = tools.selBest(pop, 1)[0]
        real_fit_tuple = evaluate_network(best_cand, args.inp, "static", gen, 0.0, args.gen) 
        if real_fit_tuple[0] < hof[0].fitness.values[0]:
            nc = toolbox.clone(best_cand)
            nc.fitness.values = real_fit_tuple
            hof.clear()
            hof.update([nc])

        best_now = hof[0]
        raw_val = best_now.fitness.values[0]
        history_log.append({'gen': gen, 'cost': raw_val/1e6})

        if gen % 10 == 0 or gen == args.gen - 1:
            try:
                cost_disp, p_disp = get_real_stats(best_now, args.inp)
                elapsed = time.time() - run_start
                status = "[OK]" if p_disp >= args.hmin else "[FAIL]"
                print(f"    [Run {run_id+1}] Gen {gen:3d}: Cost={cost_disp/1e6:.2f}M$ | P={p_disp:.2f}m {status} | Time={format_time(elapsed)}")
            except: pass

    best_ind = hof[0]
    real_cost, min_p = get_real_stats(best_ind, args.inp)
    run_time = time.time() - run_start
    print(f"    >> Run #{run_id + 1} Done ({format_time(run_time)}). Final: {real_cost/1e6:.4f} M$ | P: {min_p:.2f} m")
    
    return best_ind, real_cost, min_p, run_time, history_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default=DEFAULT_INP_FILE)
    parser.add_argument("--costs", type=str, default=DEFAULT_COST_FILE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--pop", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--gen", type=int, default=DEFAULT_GENS)
    parser.add_argument("--hmin", type=float, default=DEFAULT_H_MIN)
    parser.add_argument("--units", type=str, choices=["in", "mm"], default="mm", help="Unit system: 'in' for inches, 'mm' for millimeters")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.inp):
        print(f"Error: Input file '{args.inp}' not found!")
        sys.exit(1)

    load_config(args.costs, args.hmin, args.units)

    global_start_time = time.time()
    
    print("==============================================")
    print(f"   EVOLUTIONARY OPTIMIZER: {args.inp}")
    print("==============================================")
    print(f" Units:         {args.units}")
    print(f" Population:    {args.pop}")
    print(f" Generations:   {args.gen}")
    print(f" Min Pressure:  {args.hmin} m")
    print("==============================================")
    
    results_summary = []

    for i in range(args.runs):
        ind, cost, pressure, duration, hist = run_single_trial(i, args)
        polished_ind = deep_local_search(ind, args.inp, run_id_label=str(i+1))
        cost_pol, pressure_pol = get_real_stats(polished_ind, args.inp)
        is_feasible = (pressure_pol >= args.hmin - 0.001)
        results_summary.append({
            "run_id": i + 1, "cost": cost_pol, "pressure": pressure_pol,
            "feasible": is_feasible, "time": duration, "individual": polished_ind, "history": hist
        })

    sorted_results = sorted(results_summary, key=lambda x: (not x['feasible'], x['cost']))
    best_run = sorted_results[0]
    
    print("\n\n=== FINAL LEADERBOARD ===")
    print(f"{'Run':<4} {'Cost (M$)':<10} {'Pressure':<10} {'Time':<18} {'Status':<6}")
    print("-" * 65)
    for res in sorted_results:
        status = "OK" if res['feasible'] else "FAIL"
        print(f"{res['run_id']:<4} {res['cost']/1e6:<10.4f} {res['pressure']:<10.3f} {format_time(res['time']):<18} {status:<6}")
    
    print(f"\nCHAMPION: Run #{best_run['run_id']}")
    export_solution(best_run['individual'], best_run['history'], args.inp, "solution_champion")