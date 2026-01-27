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
#        DEFAULT CONFIGURATION (Fallback)
# ==========================================
DEFAULT_INP_FILE = "Hanoi.inp"
DEFAULT_COST_FILE = "costs.csv"
DEFAULT_POPSIZE = 200
DEFAULT_GENS = 120
DEFAULT_RUNS = 5
DEFAULT_H_MIN = 30.0

MUTATION_START = 0.25   
MUTATION_END = 0.01     
EPSILON_START = 5.0     
EPSILON_END = 0.0       

CONFIG = {
    "diameters_in": [],
    "diameters_m": [],
    "costs": {},
    "h_min": 30.0
}

# ==========================================
#           DATA MANAGEMENT
# ==========================================

def load_config(cost_file, h_min):
    if not os.path.exists(cost_file):
        print(f"[Warning] Cost file '{cost_file}' not found.")
    
    try:
        df = pd.read_csv(cost_file)
        df = df.sort_values(by="Diameter")
        
        CONFIG["diameters_in"] = df["Diameter"].tolist()
        CONFIG["diameters_m"] = [d * 0.0254 for d in df["Diameter"]] # Конвертація в метри
        CONFIG["costs"] = dict(zip(df["Diameter"], df["Cost"]))
        CONFIG["h_min"] = h_min
        
        print(f"[Config] Loaded {len(CONFIG['diameters_in'])} pipe types from {cost_file}")
        print(f"[Config] H_MIN set to {CONFIG['h_min']} m")
        
    except Exception as e:
        print(f"[Error] Failed to load config: {e}")
        sys.exit(1)

# ==========================================
#           CORE FUNCTIONS
# ==========================================

def evaluate_network(individual, inp_file, strategy="static", gen=0, tolerance=0.0, total_gens=100):
    wn = wntr.network.WaterNetworkModel(inp_file)
    pipe_names = wn.pipe_name_list
    total_cost = 0.0
    
    costs = CONFIG["costs"]
    diams_in = CONFIG["diameters_in"]
    diams_m = CONFIG["diameters_m"]
    h_min = CONFIG["h_min"]

    for i, pipe_name in enumerate(pipe_names):
        idx = individual[i]
        diam_val_in = diams_in[idx]
        wn.get_link(pipe_name).diameter = diams_m[idx]
        total_cost += wn.get_link(pipe_name).length * costs[diam_val_in]

    sim = wntr.sim.EpanetSimulator(wn)
    try:
        results = sim.run_sim()
    except Exception:
        return 1e12, 
    
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
        if strategy in ["static", "epsilon"]:
            penalty = 1e7 + (1e6 * violation)
        elif strategy == "death":
            penalty = 1e9
        elif strategy == "adaptive":
            base = 1e4 * (1.05 ** gen) 
            penalty = base + (1e5 * violation)
            if gen > total_gens * 0.75: penalty += 1e7 

    return total_cost + penalty,

def get_real_stats(individual, inp_file):
    wn = wntr.network.WaterNetworkModel(inp_file)
    total_cost = 0.0
    pipe_names = wn.pipe_name_list
    
    costs = CONFIG["costs"]
    diams_in = CONFIG["diameters_in"]
    diams_m = CONFIG["diameters_m"]
    
    for i, pipe_name in enumerate(pipe_names):
        idx = individual[i]
        diam_val_in = diams_in[idx]
        wn.get_link(pipe_name).diameter = diams_m[idx]
        total_cost += wn.get_link(pipe_name).length * costs[diam_val_in]
    
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
            change = 1 if random.random() < 0.5 else -1
            new_val = individual[i] + change
            if new_val < low: new_val = low
            elif new_val > up: new_val = up
            individual[i] = new_val
    return individual,

# --- LOCAL SEARCH MODULE ---

def simple_descent(individual, inp_file):
    curr = list(individual)
    while True:
        best_move_idx = -1
        best_cost = float('inf')
        found = False
        for i in range(len(curr)):
            if curr[i] > 0: 
                candidate = list(curr)
                candidate[i] -= 1
                res = evaluate_network(candidate, inp_file, "static", tolerance=0.0)
                cost = res[0]
                if cost < 1e7: 
                    if not found or cost < best_cost:
                        best_cost = cost
                        best_move_idx = i
                        found = True
        if found:
            curr_cost = evaluate_network(curr, inp_file, "static", tolerance=0.0)[0]
            if best_cost < curr_cost:
                 curr[best_move_idx] -= 1
            else:
                 break 
        else:
            break
    return curr

def deep_local_search(individual, inp_file, run_id_label=""):
    print(f"\n   > [Run {run_id_label}] Running Deep local search...")
    polish_start = time.time()
    
    current_best = list(individual)
    cost_tuple = evaluate_network(current_best, inp_file, "static", tolerance=0.0)
    current_cost = cost_tuple[0]
    
    num_diams = len(CONFIG["diameters_in"])
    
    improved = True
    iteration = 0
    
    while improved:
        improved = False
        iteration += 1
        
        best_move_candidate = None
        best_move_cost = current_cost
        
        for i in range(len(current_best)):
            original_diam_idx = current_best[i]
            
            if original_diam_idx > 0: 
                candidate_reduction = list(current_best)
                candidate_reduction[i] -= 1 
                
                c_tuple = evaluate_network(candidate_reduction, inp_file, "static", tolerance=0.0)
                cost = c_tuple[0]
                
                if cost < 1e7:
                    if cost < best_move_cost:
                        best_move_cost = cost
                        best_move_candidate = candidate_reduction
                else:
                    # Ремонт
                    best_repair_cost_for_i = float('inf')
                    best_repair_for_i = None
                    for j in range(len(candidate_reduction)):
                        if i == j: continue 
                        if candidate_reduction[j] < num_diams - 1: 
                            repair_candidate = list(candidate_reduction)
                            repair_candidate[j] += 1 
                            r_tuple = evaluate_network(repair_candidate, inp_file, "static", tolerance=0.0)
                            r_cost = r_tuple[0]
                            if r_cost < 1e7:
                                 if r_cost < best_repair_cost_for_i:
                                     best_repair_cost_for_i = r_cost
                                     best_repair_for_i = repair_candidate
                    if best_repair_for_i is not None:
                        if best_repair_cost_for_i < best_move_cost:
                            best_move_cost = best_repair_cost_for_i
                            best_move_candidate = best_repair_for_i
        
        if best_move_candidate is not None and best_move_cost < current_cost - 0.0001:
            current_best = best_move_candidate
            current_cost = best_move_cost
            improved = True
            
    print(f"   > Polished in {time.time() - polish_start:.1f}s. Final: {current_cost/1e6:.4f} M$")
    return current_best

# --- REPORTING & EXPORT ---

def plot_convergence_universal(history, filename="convergence_plot.png"):
    if not history: return
    gens = [entry['gen'] for entry in history]
    costs = [entry['cost'] for entry in history]
    
    min_cost = min(costs)
    view_ceiling = min_cost * 1.5
    valid_plot_costs = [c for c in costs if c < view_ceiling]
    
    if len(valid_plot_costs) < 2: valid_plot_costs = costs 
    
    y_min_limit = min(valid_plot_costs) * 0.98
    y_max_limit = max(valid_plot_costs) * 1.02
    
    plt.figure(figsize=(12, 7))
    plt.plot(gens, costs, label='Best Cost', color='blue', linewidth=2)
    plt.ylim(y_min_limit, y_max_limit)
    plt.title("Convergence Profile")
    plt.xlabel("Generation")
    plt.ylabel("Cost (M$)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def export_solution(individual, history, inp_file, filename_prefix="final_solution"):
    print(f"\n--- Generating Report ({filename_prefix}) ---")
    wn = wntr.network.WaterNetworkModel(inp_file)
    pipe_data = []
    pipe_indices = {}
    
    diams_in = CONFIG["diameters_in"]
    diams_m = CONFIG["diameters_m"]
    costs = CONFIG["costs"]
    
    for i, pipe_name in enumerate(wn.pipe_name_list):
        idx = individual[i]
        
        link = wn.get_link(pipe_name)
        start_node = link.start_node_name
        end_node = link.end_node_name
        nodes_str = f"{start_node}-{end_node}"
        
        diam_m = diams_m[idx]
        wn.get_link(pipe_name).diameter = diam_m
        
        pipe_data.append({
            "Pipe ID": pipe_name,
            "Nodes (From-To)": nodes_str,
            "Diameter (inch)": diams_in[idx],
            "Cost ($)": wn.get_link(pipe_name).length * costs[diams_in[idx]]
        })
        pipe_indices[pipe_name] = idx

    df = pd.DataFrame(pipe_data)
    total_cost = df["Cost ($)"].sum()
    df.to_csv(f"{filename_prefix}.csv", index=False)
    print(f"Table saved: {filename_prefix}.csv")
    print(f"Total Cost: {total_cost/1e6:.6f} M$")
    
    try:
        plt.figure(figsize=(14, 10))
        ax = plt.gca()
        N = len(diams_in)
        cmap = plt.get_cmap("jet", N)
        
        wntr.graphics.plot_network(wn, node_size=0, node_attribute=None, 
            link_attribute=pd.Series(pipe_indices), link_width=3.0, 
            link_cmap=cmap, link_range=[0, N-1], add_colorbar=False, ax=ax,
            title=f"Optimized Network (Cost: {total_cost/1e6:.3f} M$)")
        
        for n in wn.node_name_list:
            x, y = wn.get_node(n).coordinates
            plt.text(x, y, s=n, color='white', fontsize=8, fontweight='bold', 
                     ha='center', va='center', zorder=10,
                     bbox=dict(boxstyle="circle,pad=0.3", fc="black", ec="none", alpha=0.8))
        
        legend_patches = [mpatches.Patch(color=cmap(i/(N-1)), label=f'{d}"') for i,d in enumerate(diams_in)]
        plt.legend(handles=legend_patches, title="Diameters", loc='upper left', bbox_to_anchor=(1.01, 1))
        
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_map.png", dpi=300)
        print(f"Map saved: {filename_prefix}_map.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting map: {e}")

    if history:
        plot_convergence_universal(history, f"{filename_prefix}_plot.png")

# ==========================================
#           SINGLE TRIAL FUNCTION
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
    n_diams = len(CONFIG["diameters_in"])
    toolbox.register("attr_int", random.randint, 0, n_diams-1)
    
    wn_temp = wntr.network.WaterNetworkModel(args.inp)
    n_pipes = len(wn_temp.pipe_name_list)
    
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=n_pipes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=2)

    pop = toolbox.population(n=args.pop)
    hof = tools.HallOfFame(1)
    
    last_best_fitness = 1e9
    stagnation_counter = 0
    RESTART_THRESHOLD = 10 
    
    history_log = [] 
    
    for gen in range(args.gen):
        if gen < args.gen * 0.85: 
            progress = gen / (args.gen * 0.85)
            tol = EPSILON_START - progress * (EPSILON_START - EPSILON_END)
        else: tol = 0.0

        mut_prob = MUTATION_START - (gen/args.gen)*(MUTATION_START - MUTATION_END)
        toolbox.register("mutate", mutCreepInt, low=0, up=n_diams-1, indpb=mut_prob)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9: toolbox.mate(child1, child2); del child1.fitness.values, child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.3: toolbox.mutate(mutant); del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [evaluate_network(ind, args.inp, "epsilon", gen, tol, args.gen) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
        
        pop[:] = offspring
        
        best_cand = tools.selBest(pop, 1)[0]
        real_fit_tuple = evaluate_network(best_cand, args.inp, "static", gen, 0.0, args.gen) 
        if real_fit_tuple[0] < 1e7:
            curr_best = evaluate_network(hof[0], args.inp, "static", tolerance=0.0)[0] if len(hof)>0 else 1e9
            if real_fit_tuple[0] < curr_best:
                nc = toolbox.clone(best_cand); nc.fitness.values = real_fit_tuple; hof.clear(); hof.update([nc])
        elif len(hof) == 0: hof.update(pop)

        current_best_val = hof[0].fitness.values[0] if len(hof) > 0 else 1e9
        if abs(current_best_val - last_best_fitness) < 1000: stagnation_counter += 1
        else: stagnation_counter = 0; last_best_fitness = current_best_val

        if stagnation_counter >= RESTART_THRESHOLD and gen < args.gen - 20:
            elite = tools.selBest(pop, 1)[0]
            for i in range(len(pop)):
                if pop[i] == elite: continue 
                for gene_idx in range(len(pop[i])):
                    if random.random() < 0.30:
                        change = random.randint(-2, 2)
                        new_val = pop[i][gene_idx] + change
                        if new_val < 0: new_val = 0
                        elif new_val >= n_diams-1: new_val = n_diams-1
                        pop[i][gene_idx] = new_val
                del pop[i].fitness.values
            
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fits = [evaluate_network(ind, args.inp, "epsilon", gen, tol, args.gen) for ind in invalid_ind]
            for ind, f in zip(invalid_ind, fits): ind.fitness.values = f
            stagnation_counter = -10 

        if gen > 0 and gen % 5 == 0 and len(hof) > 0:
            imp_ind_list = simple_descent(list(hof[0]), args.inp)
            imp_ind = creator.Individual(imp_ind_list)
            imp_fit = evaluate_network(imp_ind, args.inp, "epsilon", gen, tol, args.gen)
            imp_ind.fitness.values = imp_fit
            real_imp = evaluate_network(imp_ind, args.inp, "static", tolerance=0.0)
            if real_imp[0] < evaluate_network(hof[0], args.inp, "static", tolerance=0.0)[0]:
                 nc = toolbox.clone(imp_ind); nc.fitness.values = real_imp; hof.clear(); hof.update([nc])
            pop[random.randint(0, len(pop)-1)] = imp_ind
        
        best_now = hof[0]
        raw_val = best_now.fitness.values[0]
        history_log.append({'gen': gen, 'cost': raw_val/1e6})

        if gen % 20 == 0 or gen == args.gen - 1:
            try:
                cost_disp, p_disp = get_real_stats(best_now, args.inp)
                elapsed = time.time() - run_start
                if p_disp != -999.0:
                    print(f"    [Run {run_id+1}] Gen {gen:3d}: Cost={cost_disp/1e6:.4f}M$ | P={p_disp:.2f}m | Time={elapsed:.1f}s")
                else:
                    print(f"    [Run {run_id+1}] Gen {gen:3d}: Cost(Pen)={raw_val/1e6:.4f}M$ | Time={elapsed:.1f}s")
            except: pass

    best_ind = hof[0]
    real_cost, min_p = get_real_stats(best_ind, args.inp)
    run_time = time.time() - run_start
    print(f"    >> Run #{run_id + 1} Done ({run_time:.1f}s). Final: {real_cost/1e6:.4f} M$ | P: {min_p:.2f} m")
    
    return best_ind, real_cost, min_p, run_time, history_log

# ==========================================
#           CLI & ORCHESTRATOR
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutionary Water Network Optimizer")
    parser.add_argument("--inp", type=str, default=DEFAULT_INP_FILE, help="Path to .inp file")
    parser.add_argument("--costs", type=str, default=DEFAULT_COST_FILE, help="Path to costs CSV file")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of Multi-Start runs")
    parser.add_argument("--pop", type=int, default=DEFAULT_POPSIZE, help="Population size")
    parser.add_argument("--gen", type=int, default=DEFAULT_GENS, help="Number of generations")
    parser.add_argument("--hmin", type=float, default=DEFAULT_H_MIN, help="Minimum pressure (m)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.inp):
        print(f"Error: Input file '{args.inp}' not found!")
        sys.exit(1)

    load_config(args.costs, args.hmin)

    global_start_time = time.time()
    
    print("==============================================")
    print(f"   EVOLUTIONARY OPTIMIZER: {args.inp}")
    print("==============================================")
    print(f" Configuration:")
    print(f"  - Runs:        {args.runs}")
    print(f"  - Population:  {args.pop}")
    print(f"  - Generations: {args.gen}")
    print(f"  - Pressure Min:{args.hmin} m")
    print("==============================================")
    
    results_summary = []

    for i in range(args.runs):
        ind, cost, pressure, duration, hist = run_single_trial(i, args)
        
        polished_ind = deep_local_search(ind, args.inp, run_id_label=str(i+1))
        cost_pol, pressure_pol = get_real_stats(polished_ind, args.inp)
        
        is_feasible = (pressure_pol >= args.hmin - 0.001)
        
        results_summary.append({
            "run_id": i + 1,
            "cost": cost_pol,
            "pressure": pressure_pol,
            "feasible": is_feasible,
            "time": duration,
            "individual": polished_ind,
            "history": hist
        })

    # Leaderboard: Sort by Feasibility (True first), then Cost (Low first)
    sorted_results = sorted(results_summary, key=lambda x: (not x['feasible'], x['cost']))
    best_run = sorted_results[0]
    champion_ind = best_run['individual']
    champion_hist = best_run['history']

    print("\n\n=== FINAL TOURNAMENT LEADERBOARD ===")
    print(f"{'Run':<4} {'Cost (M$)':<10} {'Pressure':<10} {'Time (s)':<8} {'Status':<6}")
    print("-" * 55)
    
    for res in sorted_results:
        marker = "(*)" if res['run_id'] == best_run['run_id'] else ""
        status = "OK" if res['feasible'] else "FAIL"
        print(f"{res['run_id']:<4} {res['cost']/1e6:<10.4f} {res['pressure']:<10.3f} {res['time']:<8.1f} {status:<6} {marker}")
    
    print(f"\nCHAMPION SELECTED: Run #{best_run['run_id']}")
    
    final_cost, final_p = get_real_stats(champion_ind, args.inp)
    
    print(f"\n=========================================")
    print(f"       FINAL OPTIMIZATION RESULTS        ")
    print(f"=========================================")
    print(f"Total Execution Time: {time.time() - global_start_time:.2f} seconds")
    print(f"Final Cost:           {final_cost/1e6:.6f} M$")
    print(f"Final Pressure:       {final_p:.3f} m")
    print(f"=========================================")
    
    export_solution(champion_ind, champion_hist, args.inp, "solution_champion")