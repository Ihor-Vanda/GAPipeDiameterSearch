import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import warnings
import wntr
from ga_config import CONFIG

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
    print("[Plot] Generating map with adaptive labels...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(inp_file)
            
        diams_raw = CONFIG["diameters_raw"]
        pipe_diams = {}
        for i, pipe_name in enumerate(wn.pipe_name_list):
            idx = individual[i]
            pipe_diams[pipe_name] = diams_raw[idx]
        
        attr_series = pd.Series(pipe_diams)

        num_nodes = len(wn.node_name_list)
        is_large_net = num_nodes > 100

        node_size = 5 if is_large_net else 20
        font_size = 4 if is_large_net else 8
        link_width = 1.5 if is_large_net else 2.5
        
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        
        unique_diams = sorted(list(set(diams_raw)))
        cmap_object = plt.get_cmap("jet")
        colors = cmap_object(np.linspace(0, 1, len(unique_diams)))
        
        wntr.graphics.plot_network(wn, link_attribute=attr_series, 
                                   node_size=node_size, 
                                   link_width=link_width, 
                                   link_cmap=cmap_object,
                                   add_colorbar=False, title=f"Optimized Solution", ax=ax)
        
        if num_nodes < 1000:
            for node_name in wn.node_name_list:
                node = wn.get_node(node_name)
                x, y = node.coordinates
                if x is not None and y is not None:
                    plt.text(x, y, s=node_name, color='white', fontsize=font_size, 
                             fontweight='bold', ha='center', va='center', zorder=10,
                             bbox=dict(boxstyle="circle,pad=0.1", fc="black", ec="none", alpha=0.6))
        
        # Легенда
        patches = [mpatches.Patch(color=colors[i], label=f"{d}") for i, d in enumerate(unique_diams)]
        plt.legend(handles=patches, title=f"Diameters ({CONFIG['unit_system']})", 
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[Plot] Saved network map: {filename}")
        
    except Exception as e:
        print(f"\n[PLOT ERROR] Failed to generate map: {e}")

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
    
    csv_name = f"{filename_prefix}.csv"
    pd.DataFrame(pipe_data).to_csv(csv_name, index=False)
    print(f"[IO] Saved table: {csv_name}")
    
    plot_convergence(history, f"{filename_prefix}_convergence.png")
    plot_network_map(individual, inp_file, f"{filename_prefix}_map.png")