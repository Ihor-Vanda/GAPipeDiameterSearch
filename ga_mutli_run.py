import os
import sys
import warnings

# --- HARD SILENCE BLOCK ---
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import wntr
import numpy as np
import pandas as pd
from deap import base, creator, tools
import random
import time
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Глушимо WNTR
wntr_logger = logging.getLogger('wntr')
wntr_logger.setLevel(logging.CRITICAL)

# ==========================================
#        DEFAULT CONFIGURATION
# ==========================================
DEFAULT_INP_FILE = "InputData/Hanoi/Hanoi.inp"
DEFAULT_COST_FILE = "InputData/Hanoi/costs.csv"
DEFAULT_POPSIZE = 100
DEFAULT_GENS = 100
DEFAULT_RUNS = 1
DEFAULT_H_MIN = 30.0

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

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    elif seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    else: return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

# ==========================================
#           DATA MANAGEMENT
# ==========================================

def load_config(cost_file, h_min, units):
    if not os.path.exists(cost_file):
        print(f"[ERROR] Cost file '{cost_file}' not found.")
        sys.exit(1)
    
    try:
        try:
            df = pd.read_csv(cost_file)
            if len(df.columns) < 2: df = pd.read_csv(cost_file, sep=';')
        except:
            df = pd.read_csv(cost_file, sep=None, engine='python')

        df.columns = df.columns.str.strip()
        diam_col = next((c for c in ["Diameter", "Diam", "D", "diameter", "size"] if c in df.columns), None)
        cost_col = next((c for c in ["Cost", "cost", "Price", "price", "UnitCost"] if c in df.columns), None)
        
        if not diam_col or not cost_col: raise KeyError("Columns Diameter/Cost not found")

        df = df.sort_values(by=diam_col)
        CONFIG["diameters_raw"] = df[diam_col].tolist()
        
        if units == "mm":
            CONFIG["diameters_m"] = [d / 1000.0 for d in df[diam_col]]
            print("[Config] Units: Millimeters (converted to meters / 1000)")
        else:
            CONFIG["diameters_m"] = [d * 0.0254 for d in df[diam_col]]
            print("[Config] Units: Inches (converted to meters * 0.0254)")
            
        CONFIG["costs"] = dict(zip(df[diam_col], df[cost_col]))
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    # 20% Max (Safety)
    for _ in range(int(pop_size * 0.2)):
        ind = [n_diams - 1] * n_pipes 
        pop.append(creator.Individual(ind))
    # 30% Median
    mid_idx = n_diams // 2
    for _ in range(int(pop_size * 0.3)):
        ind = [mid_idx] * n_pipes
        pop.append(creator.Individual(ind))
    # 50% Pure Random
    remaining = pop_size - len(pop)
    for _ in range(remaining):
        ind = [random.randint(0, n_diams-1) for _ in range(n_pipes)]
        pop.append(creator.Individual(ind))
    return pop

def run_local_search(individual, inp_file, limit_pipes=None, verbose=False):
    start_time = time.time()
    current_best = list(individual)
    cost_tuple = evaluate_network(current_best, inp_file, "static", tolerance=0.0)
    current_cost = cost_tuple[0]
    n_pipes = len(current_best)
    
    if limit_pipes is None or limit_pipes >= n_pipes:
        check_indices = list(range(n_pipes))
        mode_str = f"Full ({n_pipes})"
    else:
        check_indices = random.sample(range(n_pipes), k=limit_pipes)
        mode_str = f"Sample ({limit_pipes})"

    if verbose: print(f"   > Local Search: {mode_str}...")

    improvements = 0
    for i in check_indices:
        if current_best[i] > 0:
            candidate = list(current_best)
            candidate[i] -= 1 
            c_tuple = evaluate_network(candidate, inp_file, "static", tolerance=0.0)
            new_cost = c_tuple[0]
            if new_cost < current_cost:
                current_best = candidate
                current_cost = new_cost
                improvements += 1
                
    if verbose:
        print(f"   > Done in {format_time(time.time() - start_time)}. Imp: {improvements}. Final: {current_cost/1e6:.4f} M$")
    return current_best

# --- PLOTTING FUNCTIONS (RESTORED) ---
def plot_convergence(history, filename="convergence.png"):
    if not history: return
    gens = [x['gen'] for x in history]
    costs = [x['cost'] for x in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gens, costs, label='Cost (M$)', color='blue', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Cost (Million $)')
    plt.title('Optimization Convergence')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Plot] Saved convergence graph: {filename}")

def plot_network_map(individual, inp_file, filename="solution_map.png"):
    print("[Plot] Generating map with node labels...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(inp_file)
            
        diams_raw = CONFIG["diameters_raw"]
        
        # Підготовка даних для труб
        pipe_diams = {}
        for i, pipe_name in enumerate(wn.pipe_name_list):
            idx = individual[i]
            pipe_diams[pipe_name] = diams_raw[idx]
        
        attr_series = pd.Series(pipe_diams)

        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        unique_diams = sorted(list(set(diams_raw)))
        
        # Виправлення для нових версій matplotlib (об'єкт cmap)
        cmap_object = plt.get_cmap("jet")
        colors = cmap_object(np.linspace(0, 1, len(unique_diams)))
        
        # 1. Малюємо труби та вузли
        wntr.graphics.plot_network(wn, 
                                   link_attribute=attr_series, 
                                   node_size=20, # Трохи більші вузли
                                   link_width=2.5, 
                                   link_cmap=cmap_object,
                                   add_colorbar=False,
                                   title=f"Optimized Solution", 
                                   ax=ax)
        
        # 2. ДОДАЄМО НАЗВИ ВУЗЛІВ (Labels)
        # Адаптивний шрифт: якщо вузлів багато - шрифт менший
        num_nodes = len(wn.node_name_list)
        font_size = 8 if num_nodes < 100 else 5
        
        for node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            x, y = node.coordinates
            
            # Малюємо мітку тільки якщо є координати
            if x is not None and y is not None:
                plt.text(x, y, s=node_name, 
                         color='white', 
                         fontsize=font_size, 
                         fontweight='bold', 
                         ha='center', va='center', 
                         zorder=10, # Щоб текст був ПОВЕРХ труб
                         bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="none", alpha=0.7))
        
        # Легенда
        patches = [mpatches.Patch(color=colors[i], label=f"{d}") for i, d in enumerate(unique_diams)]
        plt.legend(handles=patches, title=f"Diameters ({CONFIG['unit_system']})", 
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[Plot] Saved network map: {filename}")
        
    except Exception as e:
        print(f"\n[PLOT ERROR] Failed to generate map.")
        print(f"Error details: {e}")
        # import traceback
        # traceback.print_exc()

def export_solution(individual, history, inp_file, filename_prefix="final_solution"):
    print(f"\n--- Saving results ---")
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
    
    # Generate Plots
    plot_convergence(history, f"{filename_prefix}_convergence.png")
    plot_network_map(individual, inp_file, f"{filename_prefix}_map.png")

# ==========================================
#           RUN LOGIC
# ==========================================

def run_single_trial(run_id, args):
    random.seed(time.time() + run_id)
    run_start = time.time()
    print(f"\n>>> Starting Run #{run_id + 1}/{args.runs}...")
    
    # --- AUTO-TUNE SETTINGS ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn_check = wntr.network.WaterNetworkModel(args.inp)
        n_pipes_total = len(wn_check.pipe_name_list)

    # Якщо мережа мала (Hanoi, 34 труби) - Агресивний режим
    if n_pipes_total < 100:
        LS_FREQ = 5          # Кожні 5 поколінь
        LS_LIMIT = None      # Перевіряти ВСІ труби
        CACHE_ENABLED = False # Кешування на малих мережах іноді заважає вийти з ям
        print(f"    [Mode] Aggressive (Small Network: {n_pipes_total} pipes)")
    else:
        # Для L-Town - Економний режим
        LS_FREQ = 20
        LS_LIMIT = 20
        CACHE_ENABLED = True
        print(f"    [Mode] Eco/Cached (Large Network: {n_pipes_total} pipes)")

    memo = {} 
    def cached_eval(individual, generation):
        if not CACHE_ENABLED:
             if generation < args.gen * 0.75: 
                progress = generation / (args.gen * 0.75)
                tol = EPSILON_START - progress * (EPSILON_START - EPSILON_END)
             else: tol = 0.0
             return evaluate_network(individual, args.inp, "epsilon", generation, tol, args.gen)

        ind_tuple = tuple(individual)
        if ind_tuple in memo: return memo[ind_tuple]
        
        if generation < args.gen * 0.75: 
            progress = generation / (args.gen * 0.75)
            tol = EPSILON_START - progress * (EPSILON_START - EPSILON_END)
        else: tol = 0.0
            
        fit = evaluate_network(individual, args.inp, "epsilon", generation, tol, args.gen)
        memo[ind_tuple] = fit
        return fit

    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    n_diams = len(CONFIG["diameters_raw"])
    
    n_pipes = n_pipes_total
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("attr_int", random.randint, 0, n_diams-1)

    pop = create_mixed_population(n_pipes, n_diams, args.pop)
    
    if run_id == 0:
        max_ind = [n_diams - 1] * n_pipes
        _, max_p = get_real_stats(max_ind, args.inp)
        print(f"    [INFO] Feasibility Check (All Max Pipes): P={max_p:.2f} m")
        if max_p < args.hmin:
            print(f"    [CRITICAL WARNING] Target {args.hmin}m is physically IMPOSSIBLE.")
            print(f"    The best possible pressure is {max_p:.2f} m.")
    
    hof = tools.HallOfFame(1)
    history_log = [] 
    
    for ind in pop: ind.fitness.values = cached_eval(ind, 0)
    hof.update(pop)

    for gen in range(args.gen):
        mut_prob = MUTATION_START - (gen/args.gen)*(MUTATION_START - MUTATION_END)
        toolbox.register("mutate", mutCreepInt, low=0, up=n_diams-1, indpb=mut_prob)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8: toolbox.mate(child1, child2); del child1.fitness.values, child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.35: toolbox.mutate(mutant); del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind: ind.fitness.values = cached_eval(ind, gen)
        
        pop[:] = offspring
        
        # --- ADAPTIVE LOCAL SEARCH ---
        if gen > 0 and gen % LS_FREQ == 0 and len(hof) > 0:
            imp_ind_list = run_local_search(list(hof[0]), args.inp, limit_pipes=LS_LIMIT, verbose=False)
            imp_ind = creator.Individual(imp_ind_list)
            imp_fit = cached_eval(imp_ind, gen)
            imp_ind.fitness.values = imp_fit
            
            if imp_fit[0] < hof[0].fitness.values[0]:
                hof.clear()
                hof.update([imp_ind])
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
    parser.add_argument("--units", type=str, choices=["in", "mm"], default="mm")
    
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
        
        # Smart Polish: All pipes for small nets, 500 for large
        wn_tmp = wntr.network.WaterNetworkModel(args.inp)
        polish_limit = None if len(wn_tmp.pipe_name_list) < 200 else 500
        
        polished_ind = run_local_search(ind, args.inp, limit_pipes=polish_limit, verbose=True)
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