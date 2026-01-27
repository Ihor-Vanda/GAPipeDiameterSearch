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

# ==========================================
#              CONFIGURATION
# ==========================================
INP_FILE = "Hanoi.inp"

POPULATION_SIZE = 200   
GENERATIONS = 100 

H_MIN = 30.0            

MUTATION_START = 0.25   
MUTATION_END = 0.01     

# Epsilon
EPSILON_START = 5.0     
EPSILON_END = 0.0       

AVAILABLE_DIAMETERS = [12.0, 16.0, 20.0, 24.0, 30.0, 40.0]
DIAMETERS_M = [d * 0.0254 for d in AVAILABLE_DIAMETERS]
COSTS = {
    12.0: 45.73, 16.0: 70.40, 20.0: 98.39, 
    24.0: 129.33, 30.0: 180.75, 40.0: 278.28
}

# ==========================================
#           CORE FUNCTIONS
# ==========================================

def evaluate_network(individual, strategy="static", gen=0, tolerance=0.0):
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    pipe_names = wn.pipe_name_list
    total_cost = 0.0
    
    for i, pipe_name in enumerate(pipe_names):
        idx = individual[i]
        diam_in = AVAILABLE_DIAMETERS[idx]
        wn.get_link(pipe_name).diameter = DIAMETERS_M[idx]
        total_cost += wn.get_link(pipe_name).length * COSTS[diam_in]

    sim = wntr.sim.EpanetSimulator(wn)
    try:
        results = sim.run_sim()
    except Exception:
        return 1e9, 
    
    pressures = results.node['pressure'].iloc[-1]
    junctions = wn.junction_name_list
    
    violation = 0.0
    effective_limit = H_MIN - tolerance
    
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
            if gen > GENERATIONS * 0.75: penalty += 1e7 

    return total_cost + penalty,

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

def simple_descent(individual):
    curr = list(individual)
    while True:
        best_move_idx = -1
        best_cost = float('inf')
        found = False
        
        for i in range(len(curr)):
            if curr[i] > 0: 
                candidate = list(curr)
                candidate[i] -= 1
                res = evaluate_network(candidate, "static", tolerance=0.0)
                cost = res[0]
                if cost < 1e7: 
                    if not found or cost < best_cost:
                        best_cost = cost
                        best_move_idx = i
                        found = True
        
        if found:
            curr_cost = evaluate_network(curr, "static", tolerance=0.0)[0]
            if best_cost < curr_cost:
                 curr[best_move_idx] -= 1
            else:
                 break 
        else:
            break
    return curr

def deep_local_search(individual):
    print("\n   > Running FINAL POLISHING (Global Best Strategy)...")
    
    current_best = list(individual)
    cost_tuple = evaluate_network(current_best, "static", tolerance=0.0)
    current_cost = cost_tuple[0]
    print(f"      Start Cost: {current_cost/1e6:.4f} M$")
    
    improved = True
    iteration = 0
    
    while improved:
        improved = False
        iteration += 1
        
        best_move_candidate = None
        best_move_cost = current_cost
        
        for i in range(len(current_best)):
            original_diam_idx = current_best[i]
            
            if original_diam_idx >= 2: 
                candidate_reduction = list(current_best)
                candidate_reduction[i] -= 1 
                
                c_tuple = evaluate_network(candidate_reduction, "static", tolerance=0.0)
                cost = c_tuple[0]
                
                if cost < 1e7:
                    if cost < best_move_cost:
                        best_move_cost = cost
                        best_move_candidate = candidate_reduction
                
                else:
                    best_repair_for_i = None
                    best_repair_cost_for_i = float('inf')
                    
                    for j in range(len(candidate_reduction)):
                        if i == j: continue 
                        if candidate_reduction[j] < len(AVAILABLE_DIAMETERS) - 1: 
                            repair_candidate = list(candidate_reduction)
                            repair_candidate[j] += 1 
                            
                            r_tuple = evaluate_network(repair_candidate, "static", tolerance=0.0)
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
            print(f"      [Iter {iteration}] Improved! Cost: {current_cost/1e6:.4f} -> {best_move_cost/1e6:.4f} M$")
            current_best = best_move_candidate
            current_cost = best_move_cost
            improved = True
        else:
            print("      No further improvements found.")

    return current_best

# --- REPORTING ---

def export_solution(individual, filename_prefix="final_solution"):
    print(f"\n--- Generating Report ({filename_prefix}) ---")
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    pipe_data = []
    pipe_indices = {}
    
    for i, pipe_name in enumerate(wn.pipe_name_list):
        idx = individual[i]
        diam_m = DIAMETERS_M[idx]
        wn.get_link(pipe_name).diameter = diam_m
        pipe_data.append({
            "Pipe ID": pipe_name,
            "Diameter (inch)": AVAILABLE_DIAMETERS[idx],
            "Cost ($)": wn.get_link(pipe_name).length * COSTS[AVAILABLE_DIAMETERS[idx]]
        })
        pipe_indices[pipe_name] = idx

    df = pd.DataFrame(pipe_data)
    total_cost = df["Cost ($)"].sum()
    df.to_csv(f"{filename_prefix}.csv", index=False)
    print(f"Total Cost: {total_cost/1e6:.6f} M$")

    try:
        plt.figure(figsize=(14, 10))
        ax = plt.gca()
        N = len(AVAILABLE_DIAMETERS)
        cmap = plt.get_cmap("jet", N)
        wntr.graphics.plot_network(wn, node_size=0, node_attribute=None, 
            link_attribute=pd.Series(pipe_indices), link_width=2.5, 
            link_cmap=cmap, link_range=[0, N-1], add_colorbar=False, ax=ax,
            title=f"Optimized Network (Cost: {total_cost/1e6:.3f} M$)")
        for n in wn.node_name_list:
            x, y = wn.get_node(n).coordinates
            plt.text(x, y, s=n, color='white', fontsize=8, fontweight='bold', 
                     ha='center', va='center', zorder=10,
                     bbox=dict(boxstyle="circle,pad=0.3", fc="black", ec="none", alpha=0.8))
        plt.legend(handles=[mpatches.Patch(color=cmap(i/(N-1)), label=f'{d}"') for i,d in enumerate(AVAILABLE_DIAMETERS)],
                   loc='upper left', bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_map.png", dpi=300)
        plt.close()
    except Exception: pass

# ==========================================
#           MAIN GA EXECUTION
# ==========================================

def run_ga(strategy_name):
    print(f"\n=== Running Strategy: {strategy_name.upper()} ===")
    
    start_time = time.time()
    
    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, len(AVAILABLE_DIAMETERS)-1)
    
    wn_temp = wntr.network.WaterNetworkModel(INP_FILE)
    n_pipes = len(wn_temp.pipe_name_list)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=n_pipes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=2)

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    logbook = tools.Logbook()
    
    last_best_fitness = 1e9
    stagnation_counter = 0
    RESTART_THRESHOLD = 10
    
    with open(f"results_{strategy_name}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation", "Min Cost (M$)", "Pressure (m)", "Tol (m)"])

        for gen in range(GENERATIONS):
            if strategy_name == "epsilon":
                if gen < GENERATIONS * 0.85: 
                    progress = gen / (GENERATIONS * 0.85)
                    tol = EPSILON_START - progress * (EPSILON_START - EPSILON_END)
                else: tol = 0.0
            else: tol = 0.0

            mut_prob = MUTATION_START - (gen/GENERATIONS)*(MUTATION_START - MUTATION_END)
            toolbox.register("mutate", mutCreepInt, low=0, up=len(AVAILABLE_DIAMETERS)-1, indpb=mut_prob)

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9: 
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.3:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [evaluate_network(ind, strategy_name, gen, tol) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
            
            pop[:] = offspring
            
            best_cand = tools.selBest(pop, 1)[0]
            real_fit = evaluate_network(best_cand, "static", gen, 0.0) 
            if real_fit[0] < 1e7:
                curr_best = evaluate_network(hof[0], "static", tolerance=0.0)[0] if len(hof)>0 else 1e9
                if real_fit[0] < curr_best:
                    nc = toolbox.clone(best_cand)
                    nc.fitness.values = real_fit
                    hof.clear(); hof.update([nc])
            elif len(hof) == 0: hof.update(pop)

            current_best_val = hof[0].fitness.values[0] if len(hof) > 0 else 1e9
            if abs(current_best_val - last_best_fitness) < 1000:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = current_best_val

            if stagnation_counter >= RESTART_THRESHOLD and gen < GENERATIONS - 20:
                print(f"   [Diversity] Stagnation ({stagnation_counter} gens). HYPER-MUTATION triggered!")
                
                elite = tools.selBest(pop, 1)[0]
                
                for i in range(len(pop)):
                    if pop[i] == elite: continue 
                    
                    for gene_idx in range(len(pop[i])):
                        if random.random() < 0.30:
                            change = random.randint(-2, 2)
                            new_val = pop[i][gene_idx] + change
                            if new_val < 0: new_val = 0
                            elif new_val >= len(AVAILABLE_DIAMETERS): new_val = len(AVAILABLE_DIAMETERS)-1
                            pop[i][gene_idx] = new_val
                    
                    del pop[i].fitness.values

                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fits = [evaluate_network(ind, strategy_name, gen, tol) for ind in invalid_ind]
                for ind, f in zip(invalid_ind, fits): ind.fitness.values = f
                
                stagnation_counter = -10 

            if gen > 0 and gen % 5 == 0: 
                if len(hof) > 0:
                    imp_ind_list = simple_descent(list(hof[0]))
                    imp_ind = creator.Individual(imp_ind_list)
                    imp_fit = evaluate_network(imp_ind, strategy_name, gen, tol)
                    imp_ind.fitness.values = imp_fit
                    
                    real_imp = evaluate_network(imp_ind, "static", tolerance=0.0)
                    if real_imp[0] < evaluate_network(hof[0], "static", tolerance=0.0)[0]:
                         print(f"   [Memetic] Improved Best: {real_imp[0]/1e6:.4f} M$")
                         nc = toolbox.clone(imp_ind)
                         nc.fitness.values = real_imp
                         hof.clear(); hof.update([nc])
                    
                    pop[random.randint(0, len(pop)-1)] = imp_ind

            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
            
            try:
                best_ind_log = hof[0]
                cost_disp = evaluate_network(best_ind_log, "static", tolerance=0.0)[0]/1e6
                wn_check = wntr.network.WaterNetworkModel(INP_FILE)
                for i, p_name in enumerate(wn_check.pipe_name_list):
                     wn_check.get_link(p_name).diameter = DIAMETERS_M[best_ind_log[i]]
                sim_check = wntr.sim.EpanetSimulator(wn_check)
                res = sim_check.run_sim()
                min_p = res.node['pressure'].iloc[-1].loc[wn_check.junction_name_list].min()
            except: min_p, cost_disp = -999, record['min']/1e6
            
            writer.writerow([gen, f"{cost_disp:.6f}", f"{min_p:.4f}", f"{tol:.2f}"])
            if gen % 10 == 0 or gen == GENERATIONS - 1:
                print(f"Gen {gen:3d}: Cost={cost_disp:.3f}M$ | P={min_p:.2f}m | Tol={tol:.2f}m | Stag={stagnation_counter}")

    print("\n--- Applying Final Deep Optimization (Global Best Strategy) ---")
    if len(hof) > 0:
        final_ind = deep_local_search(hof[0])
    else:
        final_ind = deep_local_search(pop[0])
    
    wn_check = wntr.network.WaterNetworkModel(INP_FILE)
    for i, p_name in enumerate(wn_check.pipe_name_list):
         wn_check.get_link(p_name).diameter = DIAMETERS_M[final_ind[i]]
    sim = wntr.sim.EpanetSimulator(wn_check)
    res = sim.run_sim()
    final_p = res.node['pressure'].iloc[-1].loc[wn_check.junction_name_list].min()
    
    duration = time.time() - start_time
    print(f"Final Pressure = {final_p:.3f} m")
    print(f"Time: {duration} s")
    
    export_solution(final_ind, f"solution_{strategy_name}")
    return logbook, duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=["death", "static", "adaptive", "epsilon", "all"], default="epsilon")
    args = parser.parse_args()
    
    if args.strategy == "all":
        for s in ["static", "adaptive", "epsilon"]: run_ga(s)
    else:
        run_ga(args.strategy)