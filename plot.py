import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import wntr
import os

import netdraw
from matplotlib.ticker import FuncFormatter

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

def plot_convergence(
    history,
    filename="convergence.png",
    title="Крива збіжності",
    x_label="Кількість викликів гідравлічного симулятора",
    y_label="Вартість",
    scale_cost=True
):
    if not history:
        print("   > [WARNING] Історія порожня, графік збіжності не згенеровано.")
        return

    x_vals = []
    y_vals = []

    for point in history:
        x = point.get("evals", point.get("gen"))
        y = point.get("cost", point.get("min_cost"))

        if (x is not None and y is not None and y != float("inf") and y > 0):
            x_vals.append(x)
            y_vals.append(y)

    if not x_vals:
        print("   > [WARNING] Не знайдено валідних точок для графіка.")
        return

    sorted_pairs = sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])
    x_vals, y_vals = zip(*sorted_pairs)

    if scale_cost:
        y_plot = [v / 1e6 for v in y_vals]
        y_axis_label = f"{y_label} (×10⁶)"
    else:
        y_plot = y_vals
        y_axis_label = y_label

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    ax.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    ax.grid(True, color='gray', linestyle='--', alpha=0.5)

    ax.step(
        x_vals,
        y_plot,
        where="post",
        color="#00ffcc", 
        linewidth=2.5,
        label="Найкраще знайдене значення"
    )

    max_eval = max(x_vals)

    if max_eval >= 1e6:
        divisor = 1e6
        suffix = "млн"
    elif max_eval >= 1e3:
        divisor = 1e3
        suffix = "тис."
    else:
        divisor = 1
        suffix = None

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value / divisor:g}")
    )

    if suffix:
        ax.set_xlabel(f"{x_label}, {suffix}")
    else:
        ax.set_xlabel(x_label)

    ax.set_ylabel(y_axis_label)
    ax.set_title(title, color='black', pad=15)

    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print(f"   > ✅ Збережено графік збіжності: {filename}")

    base_dir = os.path.dirname(os.path.dirname(filename))
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    csv_x_name = "Evaluations" if "evals" in history[0] else "Generation"
    csv_y_name = "Cost" if history[0].get("cost") is not None else "Objective"

    pd.DataFrame({
        csv_x_name: x_vals,
        csv_y_name: y_vals
    }).to_csv(
        os.path.join(tables_dir, "convergence_history.csv"),
        index=False
    )


def plot_network_map(individual, inp_file, filename="solution_map.png", config=None, cost=None):
    """PNG-карта оптимізованої мережі.

    Геометрія, шкала діаметрів і вузли — спільні з GUI (`netdraw`); тут
    лишається лише те, що специфічне для звіту: розмір полотна, заголовок
    із вартістю і запис файлу.
    """
    try:
        geometry = netdraw.NetworkGeometry.load(inp_file)

        diameters = netdraw.diameters_from_indices(
            geometry, individual,
            diameters_m=getattr(config, "diameters_m", None),
            diameters_raw=getattr(config, "diameters_raw", None),
        )

        fig, ax = plt.subplots(figsize=(14, 14), dpi=300)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Звіт малює труби в порядку `pipe_name_list` — у тому ж порядку,
        # в якому індексується вектор рішення.
        scale = netdraw.draw(
            ax, geometry, diameters,
            source_size=netdraw.SOURCE_SIZE_REPORT,
            legend_labels=True, nodes_on_top=False,
        )

        units_str = getattr(config, "unit_system", "м")
        scale.attach_colorbar(fig, ax, f"Діаметр ({units_str})", pad=0.02)

        if geometry.sources:
            ax.legend(scatterpoints=1, loc="upper right", fontsize=12)

        title_str = "Оптимізована конфігурація мережі"
        if cost is not None:
            title_str += f" | Вартість: {cost/1e6:.4f} M$"
        ax.set_title(title_str, fontsize=16, color="black", pad=15)

        plt.tight_layout()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"   > ✅ Збережено графік топології: {filename}")

    except Exception as e:
        print(f"\n[PLOT ERROR] Не вдалося згенерувати карту: {e}")
        import traceback
        traceback.print_exc()
