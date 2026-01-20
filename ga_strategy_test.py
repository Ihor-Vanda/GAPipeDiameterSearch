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

# Genetic Algorithm Parameters
POPULATION_SIZE = 200   
GENERATIONS = 100       

# Constraints
H_MIN = 30.0            # Minimum required pressure (meters)

# Mutation Probabilities
P_MUTATION_IND = 0.3    # Probability of mutating an individual
P_MUTATION_GENE = 0.1   # Probability of mutating a specific gene (pipe)

# Pipe Database (Diameter in inches -> Cost in $/m)
AVAILABLE_DIAMETERS = [12.0, 16.0, 20.0, 24.0, 30.0, 40.0]
DIAMETERS_M = [d * 0.0254 for d in AVAILABLE_DIAMETERS] # Convert to meters
COSTS = {
    12.0: 45.73, 16.0: 70.40, 20.0: 98.39, 
    24.0: 129.33, 30.0: 180.75, 40.0: 278.28
}

# ==========================================
#           CORE FUNCTIONS
# ==========================================

def evaluate_network(individual, strategy="static", gen=0):
    """
    Calculates the cost of the network and applies penalties for pressure violations.
    """
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    pipe_names = wn.pipe_name_list
    total_cost = 0.0
    
    # Assign diameters from the individual (genotype) to the network model
    for i, pipe_name in enumerate(pipe_names):
        idx = individual[i]
        diam_in = AVAILABLE_DIAMETERS[idx]
        wn.get_link(pipe_name).diameter = DIAMETERS_M[idx]
        total_cost += wn.get_link(pipe_name).length * COSTS[diam_in]

    # Run Hydraulic Simulation
    sim = wntr.sim.EpanetSimulator(wn)
    try:
        results = sim.run_sim()
    except Exception:
        # If simulation fails (e.g., negative pressure), return high penalty
        return 1e9, 
    
    # Check Pressure Constraints
    pressures = results.node['pressure'].iloc[-1]
    junctions = wn.junction_name_list
    
    violation = 0.0
    for node in junctions:
        p = pressures[node]
        if p < H_MIN:
            violation += (H_MIN - p)

    # Calculate Penalty based on Strategy
    penalty = 0.0
    if violation > 0:
        if strategy == "death":
            penalty = 1e9
            
        elif strategy == "static":
            # Barrier Penalty (Recommended)
            penalty = 1e7 + (1e6 * violation)
            
        elif strategy == "adaptive":
            # Adaptive Penalty that grows with generations
            base_penalty = 1e4 * (1.05 ** gen) 
            penalty = base_penalty + (1e5 * violation)
            # Hard barrier in late generations
            if gen > GENERATIONS * 0.75:
                 penalty += 1e7 

    return total_cost + penalty,

def mutCreepInt(individual, low, up, indpb):
    """
    Custom Mutation: Changes the diameter index by +1 or -1.
    Helps in fine-tuning the solution.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            change = 1 if random.random() < 0.5 else -1
            new_val = individual[i] + change
            
            # Boundary checks
            if new_val < low:
                new_val = low
            elif new_val > up:
                new_val = up
                
            individual[i] = new_val
    return individual,

def optimize_local_search(individual):
    """
    Advanced Local Search: Steepest Descent.
    Iteratively finds the best single-pipe reduction that saves the most money
    without violating pressure constraints.
    """
    print("   > Running Steepest Descent (Deep Optimization)...")
    
    current_ind = list(individual)
    
    while True:
        best_move_idx = -1
        best_cost = float('inf')
        found_improvement = False
        
        # Check every pipe for potential reduction
        for i in range(len(current_ind)):
            original_val = current_ind[i]
            
            # If diameter is already minimum, skip
            if original_val == 0:
                continue
            
            # Try reducing this pipe
            candidate_ind = list(current_ind)
            candidate_ind[i] = original_val - 1
            
            # Check feasibility using 'static' strategy logic
            res = evaluate_network(candidate_ind, strategy="static")
            cost_with_penalty = res[0]
            
            # If feasible (penalty < barrier value of 1e7)
            if cost_with_penalty < 1e7:
                # We want the lowest cost found so far in this iteration
                if not found_improvement or cost_with_penalty < best_cost:
                    best_cost = cost_with_penalty
                    best_move_idx = i
                    found_improvement = True
        
        # Apply the best move found in this iteration
        if found_improvement:
            print(f"      Improvement! Reduced pipe #{best_move_idx}. New Cost: {best_cost/1e6:.4f} M$")
            current_ind[best_move_idx] -= 1
        else:
            print("      Local minimum reached. No further improvements possible.")
            break
            
    return current_ind

# ==========================================
#        VISUALIZATION & REPORTING
# ==========================================

def export_solution(individual, filename_prefix="final_solution"):
    """
    Generates a CSV report and a visual map.
    Style: Large black nodes with white numbers inside.
    """
    print(f"\n--- Generating Report ({filename_prefix}) ---")
    
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    pipe_data = []
    pipe_indices = {} 
    
    # Apply diameters and collect data
    for i, pipe_name in enumerate(wn.pipe_name_list):
        idx = individual[i]
        diam_inch = AVAILABLE_DIAMETERS[idx]
        diam_m = DIAMETERS_M[idx]
        
        link = wn.get_link(pipe_name)
        link.diameter = diam_m
        
        pipe_data.append({
            "Pipe ID": pipe_name,
            "Start Node": link.start_node_name,
            "End Node": link.end_node_name,
            "Length (m)": link.length,
            "Diameter (inch)": diam_inch,
            "Cost ($)": link.length * COSTS[diam_inch]
        })
        pipe_indices[pipe_name] = idx

    # Save CSV
    df = pd.DataFrame(pipe_data)
    total_cost = df["Cost ($)"].sum()
    csv_filename = f"{filename_prefix}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Table saved: {csv_filename}")
    print(f"Total Cost: {total_cost/1e6:.6f} M$")

    # Generate Map
    try:
        plt.figure(figsize=(14, 10))
        
        # Colormap setup
        N_diams = len(AVAILABLE_DIAMETERS)
        universal_cmap = plt.get_cmap("jet", N_diams)
        link_attributes = pd.Series(pipe_indices)
        
        # 1. Plot Network (Pipes & Nodes background)
        # Note: We set node_attribute=None to make them uniform color
        wntr.graphics.plot_network(
            wn, 
            node_attribute=None,      # Do not color nodes by elevation/pressure
            node_size=400,            # BIG circles to fit text
            node_alpha=1.0,           # Fully opaque to hide pipes behind
            node_labels=False,        # We will add custom labels
            title=f"Optimized Network (Cost: {total_cost/1e6:.3f} M$)",
            link_attribute=link_attributes,
            link_width=3.0,           # Slightly thicker pipes
            link_cmap=universal_cmap,
            link_range=[0, N_diams-1],
            add_colorbar=False
        )
        
        # 2. Customize Nodes (Make them black for contrast)
        # wntr returns axes, but it's easier to just plot over current axes
        ax = plt.gca()
        
        # 3. Add Labels inside nodes
        for node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            x, y = node.coordinates
            
            # Draw the node circle manually to ensure color control (Black circle)
            circle = mpatches.Circle((x, y), radius=0, color='black', zorder=2)
            ax.add_patch(circle)
            
            # Draw white text inside
            plt.text(
                x, y, 
                s=node_name, 
                color='white',      # High contrast
                fontsize=8, 
                fontweight='bold',
                ha='center',        # Horizontal center
                va='center',        # Vertical center
                zorder=3            # Top layer
            )

        # 4. Legend
        legend_patches = []
        for i, diam in enumerate(AVAILABLE_DIAMETERS):
            color = universal_cmap(i / max(1, N_diams - 1)) if N_diams > 1 else universal_cmap(0)
            label_text = f'{diam:.1f}" ({COSTS[diam]:.0f} $/m)'
            patch = mpatches.Patch(color=color, label=label_text)
            legend_patches.append(patch)
            
        plt.legend(
            handles=legend_patches, 
            title="Diameter Options", 
            loc='upper left', 
            bbox_to_anchor=(1.01, 1),
            fontsize=10
        )
        
        plt.tight_layout()
        map_filename = f"{filename_prefix}_map.png"
        plt.savefig(map_filename, dpi=300)
        print(f"Map saved: {map_filename}")
        plt.close() 
        
    except Exception as e:
        print(f"Error plotting map: {e}")

# ==========================================
#           MAIN GA EXECUTION
# ==========================================

def run_ga(strategy_name):
    print(f"\n=== Running Strategy: {strategy_name.upper()} ===")
    
    # DEAP Setup (re-defined here to be safe)
    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, len(AVAILABLE_DIAMETERS)-1)

    wn_temp = wntr.network.WaterNetworkModel(INP_FILE)
    NUM_PIPES = len(wn_temp.pipe_name_list)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=NUM_PIPES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutCreepInt, low=0, up=len(AVAILABLE_DIAMETERS)-1, indpb=P_MUTATION_GENE)
    toolbox.register("select", tools.selTournament, tournsize=2)

    # Initialization
    start_time = time.time()
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    logbook = tools.Logbook()
    
    # Prepare CSV for convergence logging
    csv_filename = f"results_{strategy_name}.csv"
    
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation", "Min Cost (M$)", "Best Min Pressure (m)"])

        # Main Evolution Loop
        for gen in range(GENERATIONS):
            # Selection & Cloning
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < P_MUTATION_IND:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [evaluate_network(ind, strategy=strategy_name, gen=gen) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Update Population & Hall of Fame
            pop[:] = offspring
            hof.update(pop)
            
            # Elitism: Ensure best solution stays
            if hof[0].fitness.values[0] < pop[0].fitness.values[0]:
                pop[0] = hof[0]

            # Logging
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
            
            best_val = hof[0].fitness.values[0] / 1e6
            
            # Calculate pressure for the best individual to log it
            min_p = -1.0
            try:
                best_ind = hof[0]
                # Re-create model to check pressure
                wn_check = wntr.network.WaterNetworkModel(INP_FILE)
                for i, p_name in enumerate(wn_check.pipe_name_list):
                     wn_check.get_link(p_name).diameter = DIAMETERS_M[best_ind[i]]
                sim_check = wntr.sim.EpanetSimulator(wn_check)
                res = sim_check.run_sim()
                min_p = res.node['pressure'].iloc[-1].loc[wn_check.junction_name_list].min()
            except:
                min_p = -999.0
            
            writer.writerow([gen, f"{best_val:.6f}", f"{min_p:.4f}"])
            
            if gen % 10 == 0 or gen == GENERATIONS - 1:
                print(f"Gen {gen:3d}: Cost = {best_val:.3f} M$ | Min Pressure = {min_p:.4f} m")

    # Post-Processing: Local Search
    print("\n--- Applying Local Search ---")
    refined_ind = optimize_local_search(hof[0])
    
    # Final check
    wn_check = wntr.network.WaterNetworkModel(INP_FILE)
    for i, p_name in enumerate(wn_check.pipe_name_list):
         wn_check.get_link(p_name).diameter = DIAMETERS_M[refined_ind[i]]
    sim = wntr.sim.EpanetSimulator(wn_check)
    res = sim.run_sim()
    final_p = res.node['pressure'].iloc[-1].loc[wn_check.junction_name_list].min()
    
    duration = time.time() - start_time
    
    print(f"Final Pressure = {final_p:.3f} m")
    
    # Export Report
    export_solution(refined_ind, filename_prefix=f"solution_{strategy_name}")
    
    return logbook, duration

# ==========================================
#              ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water Network Optimization using Genetic Algorithm")
    parser.add_argument("strategy", choices=["death", "static", "adaptive", "all"], 
                        default="static", nargs="?", help="Constraint handling strategy")
    args = parser.parse_args()

    # Check input file
    if not os.path.exists(INP_FILE):
        print(f"ERROR: File '{INP_FILE}' not found. Please ensure it is in the same directory.")
        exit(1)

    strategies_to_run = ["death", "static", "adaptive"] if args.strategy == "all" else [args.strategy]

    # Collect data for final plotting
    plot_data = []

    # NO plt.figure() HERE! We wait until plotting.

    for strat in strategies_to_run:
        log, duration = run_ga(strat)
        gen = log.select("gen")
        min_vals = [x / 1e6 for x in log.select("min")]
        plot_data.append((strat, gen, min_vals))
        print(f"Strategy {strat} completed in {duration:.2f} seconds.")

    # Generate Convergence Plot (Only ONE window at the end)
    print("\n--- Generating Convergence Plot ---")
    
    # Now we create the figure for convergence
    plt.figure(figsize=(10, 6))

    for strat, gen, min_vals in plot_data:
        plt.plot(gen, min_vals, label=f"{strat.capitalize()}")

    plt.axhline(y=6.081, color='r', linestyle='--', label='Global Optimum (6.081)', alpha=0.5)
    plt.title("Optimization Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Cost (M$)")
    plt.legend()
    plt.grid(True)
    plt.ylim(5.5, 7.5)
    
    plot_filename = f"plot_{args.strategy}.png"
    plt.savefig(plot_filename)
    print(f"Convergence plot saved as '{plot_filename}'")
    
    # Show the plot window only at the very end
    plt.show()