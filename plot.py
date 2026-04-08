import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import wntr
import networkx as nx
import os

def export_solution(individual, history, inp_file, filename_prefix, config, cost=None, time_sec=None, total_sims=None):
    print("[Output] Генерація інженерних звітів та INP-файлу...")
    
    base_dir = os.path.dirname(filename_prefix)
    os.makedirs(base_dir, exist_ok=True)
    csv_path = f"{filename_prefix}.csv"
    inp_path = os.path.join(base_dir, "optimized_network.inp")
    report_path = os.path.join(base_dir, "engineering_report.txt")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn = wntr.network.WaterNetworkModel(inp_file)
        
    pipe_data = []
    costs = config.costs
    diams_raw = config.diameters_raw
    real_diams_m = []
    
    for i, pipe_name in enumerate(wn.pipe_name_list):
        idx = int(individual[i])
        link = wn.get_link(pipe_name)
        unit_label = "mm" if config.unit_system == "mm" else "inch"
        
        d_val_m = config.diameters_m[idx] if hasattr(config, 'diameters_m') else diams_raw[idx] / 1000.0
        real_diams_m.append(d_val_m)
        link.diameter = d_val_m
        
        d_val_disp = round(diams_raw[idx], 2)
        cost_val = round(link.length * costs[diams_raw[idx]], 2)
        length_val = round(link.length, 2)
        
        pipe_data.append({
            "Pipe ID": pipe_name,
            "Start Node": link.start_node_name,
            "End Node": link.end_node_name,
            f"Diameter ({unit_label})": d_val_disp,
            "Length": length_val,
            "Cost": cost_val
        })
        
    pd.DataFrame(pipe_data).to_csv(csv_path, index=False)
    print(f"   > ✅ Збережено таблицю рішення: {csv_path}")

    wntr.network.write_inpfile(wn, inp_path)
    print(f"   > ✅ Збережено EPANET INP файл: {inp_path}")

    try:
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        
        if cost is None: cost = 0.0
        if time_sec is None: time_sec = 0.0
        if total_sims is None: total_sims = len(history) if history else 0
        
        pressures = results.node['pressure'].iloc[-1]
        demands = results.node['demand'].iloc[-1]
        velocities = results.link['velocity'].iloc[-1]
        headlosses = results.link['headloss'].iloc[-1]
        
        junction_names = wn.junction_name_list
        junction_pressures = pressures[junction_names]
        sorted_pressures = junction_pressures.sort_values()
        
        pipe_velocities = velocities[wn.pipe_name_list]
        sorted_velocities = pipe_velocities.sort_values(ascending=False)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=================================================================================\n")
            f.write("                   ДЕТАЛЬНИЙ ІНЖЕНЕРНИЙ ЗВІТ (ФІНАЛЬНЕ РІШЕННЯ)\n")
            f.write("=================================================================================\n")
            f.write(f"Фінальна вартість (Капітальні витрати) : {cost/1e6:.4f} M$\n")
            f.write(f"Час оптимізації                        : {time_sec/60:.1f} хвилин\n")
            f.write(f"Витрачено симуляцій                    : {total_sims:,}\n")
            f.write("=================================================================================\n\n")
            
            f.write("--- ТИСК У ВУЗЛАХ (Відсортовано за зростанням тиску) ---\n")
            f.write(f"Найнижчий тиск: {junction_pressures.min():.2f} м (Вузол: {junction_pressures.idxmin()})\n")
            f.write(f"Найвищий тиск:  {junction_pressures.max():.2f} м (Вузол: {junction_pressures.idxmax()})\n")
            f.write(f"Середній тиск:  {junction_pressures.mean():.2f} м\n\n")
            
            f.write(f"{'Вузол ID':<15} | {'Тиск (м)':<15} | {'Споживання (л/с)':<20} | {'Висота (м)':<15}\n")
            f.write("-" * 75 + "\n")
            for node_id, p_val in sorted_pressures.items():
                node = wn.get_node(node_id)
                elev = node.elevation if hasattr(node, 'elevation') else 0.0
                demand_lps = demands[node_id] * 1000 if node_id in demands else 0.0 
                f.write(f"{node_id:<15} | {p_val:<15.2f} | {demand_lps:<20.2f} | {elev:<15.2f}\n")
            
            f.write("\n\n")
            f.write("--- ШВИДКІСТЬ ТА ВТРАТИ В ТРУБԱХ (Відсортовано за спаданням швидкості) ---\n")
            f.write(f"Максимальна швидкість: {pipe_velocities.max():.4f} м/с (Труба: {pipe_velocities.idxmax()})\n")
            f.write(f"Мінімальна швидкість:  {pipe_velocities.min():.4f} м/с (Труба: {pipe_velocities.idxmin()})\n")
            f.write(f"Середня швидкість:     {pipe_velocities.mean():.4f} м/с\n\n")
            
            f.write(f"{'Труба ID':<15} | {'Швидкість (м/с)':<18} | {'Втрати (м)':<15} | {'Діаметр (м)':<15} | {'Довжина (м)':<15}\n")
            f.write("-" * 88 + "\n")
            for pipe_id, v_val in sorted_velocities.items():
                pipe = wn.get_link(pipe_id)
                hl_val = headlosses[pipe_id] if pipe_id in headlosses else 0.0
                f.write(f"{pipe_id:<15} | {v_val:<18.4f} | {hl_val:<15.4f} | {pipe.diameter:<15.3f} | {pipe.length:<15.1f}\n")
                
        print(f"   > ✅ Збережено інженерний звіт: {report_path}")
    except Exception as e:
        print(f"   > [Помилка] Не вдалося згенерувати INP/Звіт: {e}")

def plot_convergence(history, filename="convergence.png"):
    if not history: 
        print("   > [WARNING] Історія порожня, графік збіжності не згенеровано.")
        return
    
    x_vals = []
    y_costs = []
    
    for x in history:
        x_val = x.get('evals', x.get('gen', None))
        
        y_val = x.get('cost', x.get('min_cost', None))
        
        if x_val is not None and y_val is not None and y_val != float('inf') and y_val > 0:
            x_vals.append(x_val)
            y_costs.append(y_val / 1e6)
            
    if not x_vals:
        print("   > [WARNING] Не знайдено валідних точок для графіка.")
        return
        
    sorted_pairs = sorted(zip(x_vals, y_costs), key=lambda pair: pair[0])
    x_vals, y_costs = zip(*sorted_pairs)
    
    plt.figure(figsize=(10, 6))
    
    plt.step(x_vals, y_costs, label='Найкраща вартість (M$)', color='blue', linewidth=2, where='post')
        
    x_label = 'Кількість симуляцій (Evaluations)' if 'evals' in history[0] else 'Ітерації / Покоління'
    
    plt.xlabel(x_label)
    plt.ylabel('Вартість (Мільйони $)')
    plt.title('Історія оптимізації (Convergence)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   > ✅ Збережено графік збіжності: {filename}")
    
    base_dir = os.path.dirname(os.path.dirname(filename))
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    rounded_costs = [round(c * 1e6, 2) for c in y_costs]
    pd.DataFrame({x_label: x_vals, "Cost": rounded_costs}).to_csv(
        os.path.join(tables_dir, "convergence_history.csv"), index=False
    )

def plot_network_map(individual, inp_file, filename="solution_map.png", config=None, cost=None):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(inp_file)
            
        G = wn.get_graph()
        
        pos = {}
        for node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            if hasattr(node, 'coordinates') and node.coordinates is not None:
                pos[node_name] = node.coordinates
                
        if not pos:
            pos = nx.kamada_kawai_layout(G)
            
        edges = []
        real_diams_m = []
        for i, pipe_name in enumerate(wn.pipe_name_list):
            idx = int(individual[i])
            pipe = wn.get_link(pipe_name)
            edges.append((pipe.start_node_name, pipe.end_node_name))
            
            d_val_m = config.diameters_m[idx] if hasattr(config, 'diameters_m') else config.diameters_raw[idx] / 1000.0
            real_diams_m.append(d_val_m)
            
        max_d = max(real_diams_m)
        min_d = min(real_diams_m)
        line_widths = [1 + 4 * ((d - min_d) / (max_d - min_d + 1e-6)) for d in real_diams_m]
        
        fig, ax = plt.subplots(figsize=(14, 14))
        
        junctions = wn.junction_name_list
        reservoirs = wn.reservoir_name_list
        tanks = wn.tank_name_list
        
        nx.draw_networkx_nodes(G, pos, nodelist=junctions, node_size=15, node_color='black', alpha=0.6, label="Вузли")
        
        if reservoirs:
            nx.draw_networkx_nodes(G, pos, nodelist=reservoirs, node_size=150, node_color='blue', node_shape='s', label="Джерело (Reservoir)")
            
        if tanks:
            nx.draw_networkx_nodes(G, pos, nodelist=tanks, node_size=150, node_color='red', node_shape='^', label="Бак (Tank)")
            
        edges_draw = nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edges, edge_color=real_diams_m, 
            edge_cmap=plt.cm.viridis, width=line_widths, arrows=False
        )
        
        if isinstance(edges_draw, list):
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            sm = cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=min_d, vmax=max_d))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        else:
            cbar = fig.colorbar(edges_draw, ax=ax, shrink=0.5, pad=0.02)
            
        cbar.set_label('Діаметр труби (м)')
        
        ax.legend(scatterpoints=1, loc='upper right', fontsize=12)
        
        title_str = "Оптимізована конфігурація мережі"
        if cost is not None:
            title_str += f" | Вартість: {cost/1e6:.4f} M$"
        ax.set_title(title_str, fontsize=16)
        
        ax.axis('off')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"   > ✅ Збережено графік топології: {filename}")
        
    except Exception as e:
        print(f"\n[PLOT ERROR] Не вдалося згенерувати карту: {e}")
        import traceback
        traceback.print_exc()