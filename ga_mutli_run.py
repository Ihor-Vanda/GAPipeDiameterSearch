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
DEFAULT_INP_FILE = "EXNET_READY_TO_RUN.inp"
DEFAULT_COST_FILE = "costsEXNET.csv"
DEFAULT_POPSIZE = 100
DEFAULT_GENS = 100
DEFAULT_RUNS = 1
DEFAULT_H_MIN = 20.0

MUTATION_START = 0.40   
MUTATION_END = 0.10     
EPSILON_START = 5.0    
EPSILON_END = 0.0       

CONFIG = {
    "diameters_in": [],
    "diameters_m": [],
    "costs": {},
    "h_min": 20.0
}

# --- HELPER: Time Formatter (MOVED TO GLOBAL SCOPE) ---
# Тепер ця функція доступна всюди
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

def load_config(cost_file, h_min):
    if not os.path.exists(cost_file):
        print(f"[Warning] Cost file '{cost_file}' not found.")
    
    try:
        df = pd.read_csv(cost_file)
        df = df.sort_values(by="Diameter")
        
        CONFIG["diameters_in"] = df["Diameter"].tolist()
        CONFIG["diameters_m"] = [d * 0.0254 for d in df["Diameter"]]
        CONFIG["costs"] = dict(zip(df["Diameter"], df["Cost"]))
        CONFIG["h_min"] = h_min
        
        print(f"[Config] Loaded {len(CONFIG['diameters_in'])} pipe types.")
        
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

    # Optimised Loop
    for i, pipe_name in enumerate(pipe_names):
        idx = individual[i]
        total_cost += wn.get_link(pipe_name).length * costs[diams_in[idx]]
        wn.get_link(pipe_name).diameter = diams_m[idx]

    sim = wntr.sim.EpanetSimulator(wn)
    try:
        results = sim.run_sim()
    except Exception:
        return 1e15, # Crash Penalty
    
    # Constraints Check
    pressures = results.node['pressure'].iloc[-1]
    junctions = wn.junction_name_list
    
    violation = 0.0
    effective_limit = h_min - tolerance
    
    # Використовуємо суворий підхід для пошуку порушень
    # (Можна оптимізувати через numpy, але для сумісності лишаємо цикл)
    for node in junctions:
        p = pressures[node]
        if p < effective_limit:
            violation += (effective_limit - p)

    penalty = 0.0
    if violation > 0:
        # --- UPDATED PENALTY LOGIC ---
        # Використовуємо "Deadly Penalty" (1 мільярд).
        # Це гарантує, що будь-яке допустиме рішення (навіть найдорожче)
        # буде кращим за недопустиме.
        penalty_base = 1e9
        penalty = penalty_base + (1e6 * violation)

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
        total_cost += wn.get_link(pipe_name).length * costs[diams_in[idx]]
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

# --- POPULATION INITIALIZATION (UPDATED) ---
def create_mixed_population(n_pipes, n_diams, pop_size):
    pop = []
    # 1. 25% Максимальні діаметри (Гарантовано робочі)
    for _ in range(int(pop_size * 0.25)):
        ind = [n_diams - 1] * n_pipes 
        pop.append(creator.Individual(ind))

    # 2. 25% Середні діаметри
    mid_idx = n_diams // 2
    for _ in range(int(pop_size * 0.25)):
        ind = [mid_idx] * n_pipes
        pop.append(creator.Individual(ind))
        
    # 3. 50% Повний рандом
    remaining = pop_size - len(pop)
    for _ in range(remaining):
        ind = [random.randint(0, n_diams-1) for _ in range(n_pipes)]
        pop.append(creator.Individual(ind))
        
    return pop

# ==========================================
#       LOCAL SEARCH MODULE (RESTORED)
# ==========================================

# <--- RESTORED: simple_descent ---
def simple_descent(individual, inp_file):
    """Швидкий спуск для покращення поточного рішення."""
    curr = list(individual)
    
    # Для великих мереж обмежуємо кількість спроб
    max_checks = 100 if len(curr) > 300 else len(curr)
    indices = random.sample(range(len(curr)), k=max_checks)
    
    improved = False
    
    for i in indices:
        if curr[i] > 0:
            candidate = list(curr)
            candidate[i] -= 1
            
            # Перевіряємо, чи покращилось (вартість + штраф)
            curr_score = evaluate_network(curr, inp_file, "static", tolerance=0.0)[0]
            cand_score = evaluate_network(candidate, inp_file, "static", tolerance=0.0)[0]
            
            if cand_score < curr_score:
                curr[i] -= 1
                improved = True
                
    return curr

# <--- UPDATED: deep_local_search (Universal) ---
def deep_local_search(individual, inp_file, run_id_label=""):
    print(f"\n   > [Run {run_id_label}] Running Deep Local Search...")
    polish_start = time.time()
    
    current_best = list(individual)
    cost_tuple = evaluate_network(current_best, inp_file, "static", tolerance=0.0)
    current_cost = cost_tuple[0]
    
    n_pipes = len(current_best)
    
    # --- UNIVERSAL ADAPTATION ---
    # Якщо мережа мала (< 300 труб, як Hanoi), перевіряємо ВСІ труби.
    # Якщо мережа велика (> 300 труб, як EXNET), перевіряємо вибірку (50 труб).
    if n_pipes < 300:
        check_indices = list(range(n_pipes))
        print("     [Mode] Full Search (Small Network)")
    else:
        check_indices = random.sample(range(n_pipes), k=50)
        print(f"     [Mode] Sampling Search (Large Network: 50/{n_pipes} pipes)")

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
                
                # Допускаємо лише рішення без гігантського штрафу (тобто допустимі)
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

def plot_convergence_universal(history, filename="convergence_plot.png"):
    if not history: return
    gens = [entry['gen'] for entry in history]
    costs = [entry['cost'] for entry in history]
    plt.figure(figsize=(12, 7))
    plt.plot(gens, costs, label='Best Cost', color='blue', linewidth=2)
    plt.title("Convergence Profile")
    plt.xlabel("Generation")
    plt.ylabel("Cost (M$)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def export_solution(individual, history, inp_file, filename_prefix="final_solution"):
    print(f"\n--- Saving results to {filename_prefix}.csv ---")
    wn = wntr.network.WaterNetworkModel(inp_file)
    pipe_data = []
    costs = CONFIG["costs"]
    diams_in = CONFIG["diameters_in"]
    diams_m = CONFIG["diameters_m"]
    
    for i, pipe_name in enumerate(wn.pipe_name_list):
        idx = individual[i]
        link = wn.get_link(pipe_name)
        pipe_data.append({
            "Pipe ID": pipe_name,
            "Diameter (inch)": diams_in[idx],
            "Cost": link.length * costs[diams_in[idx]]
        })
    pd.DataFrame(pipe_data).to_csv(f"{filename_prefix}.csv", index=False)
    if history: plot_convergence_universal(history, f"{filename_prefix}_plot.png")

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
    wn_temp = wntr.network.WaterNetworkModel(args.inp)
    n_pipes = len(wn_temp.pipe_name_list)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("attr_int", random.randint, 0, n_diams-1)

    # 1. СТВОРЕННЯ ПОПУЛЯЦІЇ (SMART)
    pop = create_mixed_population(n_pipes, n_diams, args.pop)
    
    # 2. PRE-FLIGHT CHECK (Тільки для Run #1)
    if run_id == 0:
        max_ind = [n_diams - 1] * n_pipes
        _, max_p = get_real_stats(max_ind, args.inp)
        print(f"    [INFO] Feasibility Check (All Max Pipes): P={max_p:.2f} m")
        if max_p < args.hmin:
            print(f"    [CRITICAL WARNING] Target {args.hmin}m is physically IMPOSSIBLE.")
            print(f"    The best possible pressure is {max_p:.2f} m.")
    
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
        
        # --- RESTORED: SIMPLE DESCENT CALL ---
        # Викликаємо легкий пошук кожні 5 поколінь для найкращого індивіда
        if gen > 0 and gen % 5 == 0 and len(hof) > 0:
            imp_ind_list = simple_descent(list(hof[0]), args.inp)
            imp_ind = creator.Individual(imp_ind_list)
            # Переоцінюємо
            imp_fit = evaluate_network(imp_ind, args.inp, "epsilon", gen, tol, args.gen)
            imp_ind.fitness.values = imp_fit
            # Якщо він кращий, додаємо в HOF
            if imp_fit[0] < hof[0].fitness.values[0]:
                hof[0] = imp_ind # Замінюємо лідера
                # Також замінюємо випадкового в популяції
                pop[random.randint(0, len(pop)-1)] = imp_ind

        # Оновлення HOF (стандартне)
        best_cand = tools.selBest(pop, 1)[0]
        real_fit_tuple = evaluate_network(best_cand, args.inp, "static", gen, 0.0, args.gen) 
        
        if real_fit_tuple[0] < hof[0].fitness.values[0]:
            nc = toolbox.clone(best_cand)
            nc.fitness.values = real_fit_tuple
            hof.clear()
            hof.update([nc])

        # --- LOGGING ---
        best_now = hof[0]
        raw_val = best_now.fitness.values[0]
        history_log.append({'gen': gen, 'cost': raw_val/1e6})

        if gen % 10 == 0 or gen == args.gen - 1:
            try:
                cost_disp, p_disp = get_real_stats(best_now, args.inp)
                elapsed = time.time() - run_start
                status = "[OK]" if p_disp >= args.hmin else "[FAIL]"
                # UPDATED: Time formatting usage
                print(f"    [Run {run_id+1}] Gen {gen:3d}: Cost={cost_disp/1e6:.2f}M$ | P={p_disp:.2f}m {status} | Time={format_time(elapsed)}")
            except: pass

    best_ind = hof[0]
    real_cost, min_p = get_real_stats(best_ind, args.inp)
    run_time = time.time() - run_start
    print(f"    >> Run #{run_id + 1} Done ({format_time(run_time)}). Final: {real_cost/1e6:.4f} M$ | P: {min_p:.2f} m")
    
    return best_ind, real_cost, min_p, run_time, history_log

# ==========================================
#           CLI & ORCHESTRATOR
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default=DEFAULT_INP_FILE)
    parser.add_argument("--costs", type=str, default=DEFAULT_COST_FILE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--pop", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--gen", type=int, default=DEFAULT_GENS)
    parser.add_argument("--hmin", type=float, default=DEFAULT_H_MIN)
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
        
        # Polish (using Universal Deep Search)
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

    sorted_results = sorted(results_summary, key=lambda x: (not x['feasible'], x['cost']))
    best_run = sorted_results[0]
    champion_ind = best_run['individual']
    champion_hist = best_run['history']

    print("\n\n=== FINAL TOURNAMENT LEADERBOARD ===")
    print(f"{'Run':<4} {'Cost (M$)':<10} {'Pressure':<10} {'Time':<18} {'Status':<6}")
    print("-" * 65)
    
    for res in sorted_results:
        marker = "(*)" if res['run_id'] == best_run['run_id'] else ""
        status = "OK" if res['feasible'] else "FAIL"
        time_str = format_time(res['time'])
        print(f"{res['run_id']:<4} {res['cost']/1e6:<10.4f} {res['pressure']:<10.3f} {time_str:<18} {status:<6} {marker}")
    
    print(f"\nCHAMPION SELECTED: Run #{best_run['run_id']}")
    
    final_cost, final_p = get_real_stats(champion_ind, args.inp)
    total_duration = time.time() - global_start_time
    
    print(f"\n=========================================")
    print(f"       FINAL OPTIMIZATION RESULTS        ")
    print(f"=========================================")
    print(f"Total Execution Time: {format_time(total_duration)}")
    print(f"Final Cost:           {final_cost/1e6:.6f} M$")
    print(f"Final Pressure:       {final_p:.3f} m")
    print(f"=========================================")
    
    export_solution(champion_ind, champion_hist, args.inp, "solution_champion")