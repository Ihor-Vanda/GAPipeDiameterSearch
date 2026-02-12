import matplotlib
# Вмикаємо режим "без екрану" для сервера
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import warnings
import wntr
import os
from ga_config import CONFIG

def plot_convergence(history, filename="convergence.png"):
    if not history: return
    gens = [x['gen'] for x in history]
    costs = [x['cost'] / 1e6 for x in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gens, costs, label='Cost (M$)', color='blue', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Cost (Million $)')
    plt.title('Optimization Convergence')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Plot] Saved convergence graph: {filename}")

def plot_network_map(individual, inp_file, filename="solution_map.png"):
    print("[Plot] Generating network map...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(inp_file)
        
        node = wn.get_node(wn.junction_name_list[0])
        if not hasattr(node, 'coordinates') or node.coordinates is None:
            print("[PLOT WARN] INP file has no node coordinates. Skipping map generation.")
            return

        diams_raw = CONFIG["diameters_raw"]
        pipe_diams = {}
        for i, pipe_name in enumerate(wn.pipe_name_list):
            idx = individual[i]
            pipe_diams[pipe_name] = diams_raw[idx]
        
        attr_series = pd.Series(pipe_diams)
        
        plt.figure(figsize=(12, 10))
        
    
        wntr.graphics.plot_network(wn, link_attribute=attr_series, node_size=15, 
                                   link_width=2, link_cmap=plt.cm.viridis, 
                                   add_colorbar=False, 
                                   title="Optimized Pipe Diameters")
        
        unique_diams = sorted(list(set(diams_raw)))
        import matplotlib.colors as mcolors
        
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=min(unique_diams), vmax=max(unique_diams))
        
        patches = []
        for d in unique_diams:
            color = cmap(norm(d))
            patches.append(mpatches.Patch(color=color, label=f"{d}"))
            
        plt.legend(handles=patches, title=f"Diameters ({CONFIG['unit_system']})", 
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[Plot] Saved network map: {filename}")
        
    except Exception as e:
        print(f"\n[PLOT ERROR] Failed to generate map: {e}")

def export_solution(individual, history, inp_file, filename_prefix="final_solution"):
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
        
        d_val = diams_raw[idx]
        cost_val = link.length * costs[d_val]
        
        pipe_data.append({
            "Pipe ID": pipe_name,
            "Start Node": link.start_node_name,
            "End Node": link.end_node_name,
            f"Diameter ({unit_label})": d_val,
            "Length": link.length,
            "Cost": cost_val
        })
    
    if not filename_prefix.endswith('.csv'):
        filename = f"{filename_prefix}.csv"
    else:
        filename = filename_prefix
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    pd.DataFrame(pipe_data).to_csv(filename, index=False)
    print(f"[Output] Saved solution table: {filename}")