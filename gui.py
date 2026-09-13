import warnings

warnings.filterwarnings("ignore", message="Ignoring fixed .* limits")

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import threading
import sys
import os
import time
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import netdraw
from analytical_solver.worker import clean_temp_root, get_temp_root
from runner import OptimizationRunner, RunRequest

# ==========================================
# 1 КЛАС GUI ТА ПЕРЕНАПРАВЛЕННЯ ЛОГІВ
# ==========================================
class GUIStream:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.log_file = None

    def set_file(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.log_file = open(file_path, "a", encoding="utf-8")

    def write(self, text):
        self.text_widget.insert("end", text)
        self.text_widget.see("end")
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()

    def flush(self):
        if self.log_file:
            self.log_file.flush()
            
    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

class PipelineOptimizerApp(ctk.CTk):
    
    # ==========================================
    # 2 App Init & Setup
    # ==========================================
    
    def __init__(self):
        super().__init__()
        self.title("Оптимізатор Водопровідних Мереж")
        self.geometry("1100x700")

        self.base_font_size = 14
        self.font_main = ctk.CTkFont(family="Arial", size=self.base_font_size)
        self.font_bold = ctk.CTkFont(family="Arial", size=self.base_font_size, weight="bold")
        self.font_title = ctk.CTkFont(family="Arial", size=self.base_font_size + 4, weight="bold")
        self.font_mono = ctk.CTkFont(family="Consolas", size=self.base_font_size)

        self.selected_inp = None
        self.selected_costs = None
        self.optimization_thread = None
        self.stop_event = threading.Event()
        self.progress_queue = multiprocessing.Queue()
        self.manager = multiprocessing.Manager()
        self.shared_progress = self.manager.dict()

        self.opt_process = None

        self.history_sims = []
        self.history_costs = []

        # Стан прогону. Раніше ці поля з'являлися лише в start_thread /
        # run_optimization, тому stop_thread і reset_ui читали їх через
        # getattr або падали з AttributeError, якщо порядок натискань
        # відрізнявся від очікуваного.
        self.is_stopped = False
        self.last_run_dir = None
        self.highest_sims = 0

        # Кеш живої карти мережі.
        self._last_drawn_cost_for_map = None
        self._last_drawn_sol = None
        self._live_inp_path = None
        self.topo_cb = None

        # Вікно результатів.
        self.available_logs = {}


        self.setup_treeview_style() 
        self._setup_ui()
        
    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(1, weight=1)

        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.grid(row=0, column=0, pady=10, sticky="ew")
        
        ctk.CTkLabel(header_frame, text="Налаштування", font=self.font_title).pack(side="left", padx=10)
        
        ctk.CTkButton(header_frame, text="A-", width=30, font=self.font_bold, command=lambda: self.change_font_size(-1)).pack(side="right", padx=5)
        ctk.CTkButton(header_frame, text="A+", width=30, font=self.font_bold, command=lambda: self.change_font_size(1)).pack(side="right")

        self.tabs = ctk.CTkTabview(self.sidebar)
        self.tabs.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        tab_main = self.tabs.add("Головні")
        tab_algo = self.tabs.add("Алгоритм")

        self.btn_inp = ctk.CTkButton(tab_main, text="Вибрати .inp", font=self.font_main, command=self.select_inp)
        self.btn_inp.pack(pady=10, fill="x")
        self.btn_costs = ctk.CTkButton(tab_main, text="Вибрати ціни (.csv)", font=self.font_main, command=self.select_costs)
        self.btn_costs.pack(pady=10, fill="x")

        ctk.CTkLabel(tab_main, text="Режим запуску:", font=self.font_main).pack(anchor="w", pady=(10,0))
        self.opt_mode = ctk.CTkOptionMenu(tab_main, values=["Швидкий Аналітичний", "Повний Аналітичний"], font=self.font_main, command=self.on_mode_change)
        self.opt_mode.set("Швидкий Аналітичний")
        self.opt_mode.pack(fill="x", pady=5)

        ctk.CTkLabel(tab_main, text="Кількість запусків (runs):", font=self.font_main).pack(anchor="w", pady=(5,0))
        self.ent_runs = ctk.CTkEntry(tab_main, font=self.font_main)
        self.ent_runs.insert(0, "1")
        self.ent_runs.pack(fill="x", pady=5)
        
        ctk.CTkLabel(tab_main, text="Мін. тиск hmin (м):", font=self.font_main).pack(anchor="w", pady=(5,0))
        self.ent_hmin = ctk.CTkEntry(tab_main, font=self.font_main)
        self.ent_hmin.insert(0, "30.0")
        self.ent_hmin.pack(fill="x", pady=5)

        ctk.CTkLabel(tab_main, text="Одиниці діаметрів:", font=self.font_main).pack(anchor="w", pady=(5,0))
        self.opt_units = ctk.CTkOptionMenu(tab_main, values=["mm", "in"], font=self.font_main)
        self.opt_units.set("mm")
        self.opt_units.pack(fill="x", pady=5)

        self.lbl_cores = ctk.CTkLabel(tab_algo, text="Острови (cores, реком. >= 5):", font=self.font_main)
        self.lbl_cores.pack(anchor="w", pady=(5,0))
        self.ent_cores = ctk.CTkEntry(tab_algo, font=self.font_main)
        self.ent_cores.insert(0, "5")
        self.ent_cores.pack(fill="x", pady=5)
        
        ctk.CTkLabel(tab_algo, text="Макс симуляцій (max_sims):", font=self.font_main).pack(anchor="w", pady=(5,0))
        self.ent_sims = ctk.CTkEntry(tab_algo, font=self.font_main)
        self.ent_sims.insert(0, "3000000")
        self.ent_sims.pack(fill="x", pady=5)

        ctk.CTkLabel(tab_algo, text="Опт. швидкість v_opt (м/с):", font=self.font_main).pack(anchor="w", pady=(5,0))
        self.ent_vopt = ctk.CTkEntry(tab_algo, font=self.font_main)
        self.ent_vopt.insert(0, "1.0")
        self.ent_vopt.pack(fill="x", pady=5)

        self.btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.btn_frame.grid(row=2, column=0, padx=10, pady=20, sticky="ew")
        self.btn_frame.grid_columnconfigure((0,1), weight=1)

        self.btn_run = ctk.CTkButton(self.btn_frame, text="ЗАПУСК", fg_color="green", height=40, font=self.font_bold, command=self.start_thread)
        self.btn_run.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.btn_stop = ctk.CTkButton(self.btn_frame, text="ЗУПИНИТИ", fg_color="#b30000", hover_color="#ff3333", state="disabled", height=40, font=self.font_bold, command=self.stop_thread)
        self.btn_stop.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        self.btn_open_history = ctk.CTkButton(self.btn_frame, text="Відкрити минулий запуск", font=self.font_main, command=self.open_past_run, fg_color="transparent", border_width=2)
        self.btn_open_history.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="ew")

        self.btn_view_results = ctk.CTkButton(self.sidebar, text="📊 ПЕРЕГЛЯНУТИ РЕЗУЛЬТАТИ", state="disabled", height=40, font=self.font_bold, command=self.open_results_viewer)
        self.btn_view_results.grid(row=3, column=0, padx=10, pady=(0, 20), sticky="ew")

        self.main_view = ctk.CTkFrame(self)
        self.main_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_view.grid_rowconfigure(0, weight=3) 
        self.main_view.grid_rowconfigure(1, weight=2) 
        self.main_view.grid_columnconfigure(0, weight=1)

        self.main_tabs = ctk.CTkTabview(self.main_view)
        self.main_tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.tab_convergence = self.main_tabs.add("Графік Збіжності")
        self.tab_topology = self.main_tabs.add("Прев'ю Мережі (INP)")
        self.tab_costs = self.main_tabs.add("Таблиця Цін (CSV)")

        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.set_xlabel("Симуляції")
        self.ax.set_ylabel("Вартість (М$)")
        self.ax.grid(True, color='gray', linestyle='--', alpha=0.5)
        self.line, = self.ax.step([], [], color='#00ffcc', linewidth=2, where='post')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_convergence)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.tree_costs_preview = ttk.Treeview(self.tab_costs)
        self.tree_costs_preview.pack(fill="both", expand=True)

        self.log_frame = ctk.CTkFrame(self.main_view)
        self.log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(1, weight=1)

        self.dash_frame = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        self.dash_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.dash_frame.grid_columnconfigure(1, weight=1)
        
        self.lbl_cost = ctk.CTkLabel(self.dash_frame, text="Найкраща вартість: ---", font=self.font_bold)
        self.lbl_cost.grid(row=0, column=0, sticky="w")
        self.lbl_sims = ctk.CTkLabel(self.dash_frame, text="Симуляцій: 0", font=self.font_main)
        self.lbl_sims.grid(row=0, column=1, sticky="e")

        self.txt_logs = ctk.CTkTextbox(self.log_frame, font=self.font_mono)
        self.txt_logs.grid(row=1, column=0, sticky="nsew")
        self.txt_logs.bind("<Key>", self.prevent_typing)

        self.progress = ctk.CTkProgressBar(self.log_frame)
        self.progress.grid(row=2, column=0, pady=10, sticky="ew")
        self.progress.set(0)
        
    def prevent_typing(self, event):
        if event.state & 4 or event.state & 8: 
            if event.keysym.lower() == "c": return None 
        if event.keysym in ("Up", "Down", "Left", "Right", "Prior", "Next", "Shift_L", "Shift_R"):
            return None
        return "break"

    def change_font_size(self, delta):
        self.base_font_size += delta
        self.base_font_size = max(10, min(self.base_font_size, 24))
        
        self.font_main.configure(size=self.base_font_size)
        self.font_bold.configure(size=self.base_font_size)
        self.font_title.configure(size=self.base_font_size + 4)
        self.font_mono.configure(size=self.base_font_size)
        
    # ==========================================
    # 3 User Events
    # ==========================================
    
    def select_inp(self):
        file_path = filedialog.askopenfilename(title="Вибрати INP файл", filetypes=[("INP files", "*.inp")])
        if file_path:
            self.selected_inp = file_path
            self.btn_inp.configure(text=os.path.basename(file_path))
            self.main_tabs.set("Прев'ю Мережі (INP)")
            self._draw_basic_topology(file_path)

    def select_costs(self):
        file_path = filedialog.askopenfilename(title="Вибрати CSV з цінами", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.selected_costs = file_path
            self.btn_costs.configure(text=os.path.basename(file_path))
            self.main_tabs.set("Таблиця Цін (CSV)")
            try:
                df = pd.read_csv(file_path)
                self.load_df_to_treeview(self.tree_costs_preview, df)
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося прочитати CSV: {e}")
                
    def on_mode_change(self, _selected_mode):
        self.lbl_cores.configure(text="Острови (cores, реком. >= 5):")
        if self.ent_cores.get() == "0":
            self.ent_cores.delete(0, "end")
            self.ent_cores.insert(0, "5")
            
    # ==========================================
    # 4 Execution/Threading
    # ==========================================
    
    def stop_thread(self):
        if self.is_stopped: return
        self.is_stopped = True
        self.btn_stop.configure(state="disabled")
        print("\n[System] 🛑 Команда на зупинку. Очікуємо збереження результатів...")

    def start_thread(self):
        if not self.selected_inp or not self.selected_costs:
            messagebox.showerror("Помилка", "Виберіть INP та CSV файли.")
            return

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_view_results.configure(state="disabled")
        self.is_stopped = False
        self.highest_sims = 0
        self.txt_logs.delete("1.0", "end")
        
        self.clear_realtime_graph()
        
        sys.stdout = GUIStream(self.txt_logs)
        
        worker = threading.Thread(target=self.run_optimization)
        worker.daemon = True
        worker.start()

    def run_optimization(self):
        """Сценарій запуску тепер живе в `runner.OptimizationRunner`.

        Тут лишається лише те, що справді специфічне для GUI: зчитати
        поля вводу, підготувати каталог і лог, віддати прогрес у дашборд
        і повідомити про завершення. До етапу 5 цей метод дублював 171
        рядок сценарію з `main()`.
        """
        base_dir = None
        try:
            clean_temp_root()
            os.makedirs(get_temp_root(), exist_ok=True)

            request = self._collect_request()

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_dir = os.path.abspath(os.path.join("OutputDataExperiments", timestamp))
            self.last_run_dir = base_dir
            os.makedirs(base_dir, exist_ok=True)

            log_file_path = os.path.join(base_dir, "logs", f"run_{timestamp}.txt")
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            if isinstance(sys.stdout, GUIStream):
                sys.stdout.set_file(log_file_path)

            print(f"[System] Log file initiated: {log_file_path}")
            print("==============================================")
            print(f"   ANALYTICAL OPTIMIZER: {os.path.basename(request.inp_file)}")
            print(f"   Режим: {request.run_mode.upper()} | Воркери: {request.effective_cores}")
            print("==============================================\n")

            runner = OptimizationRunner(
                request,
                on_progress=self.sync_ui_state,
                should_stop=lambda: self.is_stopped,
                on_run_start=self._on_run_start,
            )
            runner.prepare()

            results = runner.run(base_dir)
            if runner.export(results, base_dir):
                if self.is_stopped:
                    print("\n✅ ОПТИМІЗАЦІЮ ПЕРЕРВАНО. ДАНІ ЗБЕРЕЖЕНО.")
                else:
                    print("\n✅ ОПТИМІЗАЦІЮ ЗАВЕРШЕНО! Результати збережено.")
            else:
                print("\n[System] 🛑 Процес зупинено. Даних для збереження немає.")

        except Exception as e:
            if self.is_stopped:
                print("\n[System] 🛑 Процес примусово зупинено.")
            else:
                print(f"\n❌ КРИТИЧНА ПОМИЛКА: {e}")
                import traceback
                print(traceback.format_exc())
        finally:
            try:
                import gc
                gc.collect()
                time.sleep(1.0)
                clean_temp_root()
            except Exception:
                pass

            if isinstance(sys.stdout, GUIStream):
                sys.stdout.close()

            self.after(0, self.reset_ui)

    def _collect_request(self):
        """Зчитати поля вводу в `RunRequest`.

        Розбір толерантний до порожніх і некоректних значень — це робить
        `RunRequest.from_text_fields`, і саме тому його можна перевірити
        тестами без запуску tkinter.
        """
        mode_text = self.opt_mode.get()
        return RunRequest.from_text_fields(
            inp_file=self.selected_inp,
            cost_file=self.selected_costs,
            hmin=self.ent_hmin.get(),
            units=self.opt_units.get(),
            v_opt=self.ent_vopt.get(),
            runs=self.ent_runs.get(),
            cores=self.ent_cores.get(),
            max_sims=self.ent_sims.get(),
            run_mode="fast_analytical" if mode_text == "Швидкий Аналітичний" else "analytical",
        )

    def _on_run_start(self, _run_id):
        """Перед кожним прогоном: очистити графік збіжності."""
        self.after(0, self.clear_realtime_graph)
        self.highest_sims = 0
    
    # ==========================================
    # 5 Real-time Updates
    # ==========================================
    
    def _update_dashboard(self, sims, best_cost, best_sol=None):
        if best_cost != float('inf'):
            self.lbl_cost.configure(text=f"Найкраща вартість: {best_cost/1e6:.4f} M$")
            
            if best_sol is not None:
                self._update_live_network_graph(best_sol, best_cost)
            
        if sims < self.highest_sims:
            sims = self.highest_sims
        else:
            self.highest_sims = sims
            
        self.lbl_sims.configure(text=f"Симуляцій: {sims:,}")
        
        try: max_sims = int(self.ent_sims.get())
        except ValueError: max_sims = 0
            
        if max_sims > 0:
            self.progress.set(min(1.0, sims / max_sims))
            
        if best_cost != float('inf'):
            self.history_sims.append(sims)
            self.history_costs.append(best_cost / 1e6)
            
            sorted_data = sorted(zip(self.history_sims, self.history_costs), key=lambda x: x[0])
            self.history_sims = [x[0] for x in sorted_data]
            self.history_costs = [x[1] for x in sorted_data]
            
            self.line.set_data(self.history_sims, self.history_costs)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
        
            
    def sync_ui_state(self, sims, best_cost, best_sol=None):
        if self.is_stopped:
            raise KeyboardInterrupt("Виконання перервано користувачем")
        self.after(0, self._update_dashboard, sims, best_cost, best_sol)

    def clear_realtime_graph(self):
        self.history_sims.clear()
        self.history_costs.clear()
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        
    def reset_ui(self):
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.progress.set(1 if not self.is_stopped else 0)
        
        if self.last_run_dir and os.path.exists(os.path.join(self.last_run_dir, "tables")):
            self.btn_view_results.configure(state="normal")
            
        sys.stdout = sys.__stdout__
        
    def _update_live_network_graph(self, best_sol, current_cost):
        if not self.selected_inp or not best_sol: return
        
        if self._last_drawn_cost_for_map == current_cost:
            return

        try:
            if not hasattr(self, 'topo_canvas') or not self.topo_canvas.get_tk_widget().winfo_exists():
                self._draw_basic_topology(self.selected_inp)
                
            if not hasattr(self, 'topo_ax') or not hasattr(self, 'topo_canvas'):
                return

            self._last_drawn_cost_for_map = current_cost
            self._last_drawn_sol = list(best_sol)

            # Геометрія мережі кешується, поки не змінився .inp: перечитувати
            # її на кожне оновлення прогресу було б дорожче за саме малювання.
            if self._live_inp_path != self.selected_inp:
                self._live_geometry = netdraw.NetworkGeometry.load(self.selected_inp)
                self._live_inp_path = self.selected_inp

            diameters = netdraw.diameters_from_solution(
                self._live_geometry, best_sol, self._get_actual_diams())

            self.topo_ax.clear()
            scale = netdraw.draw(self.topo_ax, self._live_geometry, diameters)

            self.topo_cb = scale.attach_colorbar(
                self.topo_fig, self.topo_ax, f"Діаметр ({self.opt_units.get()})",
                existing=self.topo_cb,
            )

            self.topo_canvas.draw()
            self.topo_canvas.flush_events()
            self.tab_topology.update_idletasks()
                   
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            if hasattr(self, 'txt_logs'):
                self.txt_logs.insert("end", f"\n❌ [GUI ERROR] Помилка малювання:\n{err_msg}\n")
                self.txt_logs.see("end")
        
    # ==========================================
    # 6 Utilities
    # ==========================================
    
    def setup_treeview_style(self):
        style = ttk.Style(self)
        
        try:
            style.theme_use("default")
        except AttributeError:
            pass
        
        bg_color = "#2b2b2b"    
        fg_color = "#ffffff"    
        selected_bg = "#1f538d"
        heading_bg = "#333333"  
        
        style.configure("Treeview",
                        background=bg_color,
                        foreground=fg_color,
                        rowheight=25,
                        fieldbackground=bg_color,
                        borderwidth=0)
        
        style.map('Treeview', background=[('selected', selected_bg)])
        
        style.configure("Treeview.Heading",
                        background=heading_bg,
                        foreground=fg_color,
                        relief="flat",
                        font=("Arial", 11, "bold"))
        
        style.map("Treeview.Heading", background=[('active', '#444444')])

    def load_df_to_treeview(self, tree, df):
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        
        for col in tree["columns"]:
            tree.heading(col, text=col, command=lambda c=col: self.treeview_sort_column(tree, c, False))
            tree.column(col, width=100, anchor="center")
            
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
            
    def treeview_sort_column(self, tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        try:
            l.sort(key=lambda t: float(t[0]) if t[0] and t[0] != 'nan' else -float('inf'), reverse=reverse)
        except ValueError:
            l.sort(key=lambda t: str(t[0]).lower(), reverse=reverse)

        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

        tv.heading(col, command=lambda: self.treeview_sort_column(tv, col, not reverse))

    def enable_mouse_navigation(self, fig, ax, canvas):
        parent_frame = canvas.get_tk_widget().master
        
        toolbar = NavigationToolbar2Tk(canvas, parent_frame, pack_toolbar=False)
        toolbar.update()
        
        toolbar.pan()

        def zoom(event):
            if event.inaxes != ax: return
            scale = 1.2 if event.button == 'up' else 1 / 1.2
            x, y = event.xdata, event.ydata
            xlim, ylim = ax.get_xlim(), ax.get_ylim()

            ax.set_xlim([x - (x - xlim[0]) / scale, x + (xlim[1] - x) / scale])
            ax.set_ylim([y - (y - ylim[0]) / scale, y + (ylim[1] - y) / scale])
            canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', zoom)
        
    def _get_actual_diams(self):
        if not self.selected_costs or not os.path.exists(self.selected_costs):
            return None
        try:
            df = pd.read_csv(self.selected_costs)
            for col in df.columns:
                if 'diam' in col.lower() or 'діаметр' in col.lower():
                    return df[col].astype(float).tolist()
        except:
            pass
        return None
    
    def _draw_basic_topology(self, inp_path):
        for widget in self.tab_topology.winfo_children(): widget.destroy()
        
        if hasattr(self, 'topo_fig'):
            plt.close(self.topo_fig) 
            
        try:
            geometry = netdraw.NetworkGeometry.load(inp_path)

            self.topo_fig, self.topo_ax = plt.subplots(figsize=(6, 4), dpi=100)
            self.topo_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            bg_color = '#ffffff'
            self.topo_fig.patch.set_facecolor(bg_color)
            self.topo_ax.set_facecolor(bg_color)

            scale = netdraw.draw(self.topo_ax, geometry, netdraw.diameters_from_inp(geometry))
            self.topo_cb = scale.attach_colorbar(
                self.topo_fig, self.topo_ax,
                f"Початковий Діаметр ({self.opt_units.get()})",
            )

            self.topo_canvas = FigureCanvasTkAgg(self.topo_fig, master=self.tab_topology)
            
            def on_resize(_event):
                self.topo_ax.autoscale_view(tight=True)
                self.topo_fig.canvas.draw_idle()
            self.topo_fig.canvas.mpl_connect('resize_event', on_resize)
            
            self.topo_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.topo_canvas.draw()
            self.enable_mouse_navigation(self.topo_fig, self.topo_ax, self.topo_canvas) 
            
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg)
            ctk.CTkLabel(self.tab_topology, text=f"Не вдалося побудувати граф:\n{e}", font=self.font_main).pack(pady=20)
    
    # ==========================================
    # 7 Results Viewer
    # ==========================================

    def open_past_run(self):
        directory = filedialog.askdirectory(title="Оберіть папку експерименту (напр. 2026-04-06_15-36-30)")
        
        if directory:
            has_tables = os.path.exists(os.path.join(directory, "tables"))
            has_runs = os.path.exists(os.path.join(directory, "run_1"))
            
            if has_tables or has_runs:
                self.open_results_viewer(target_dir=directory)
            else:
                messagebox.showwarning(
                    "Невірний формат", 
                    "У цій папці не знайдено результатів оптимізації.\nОберіть папку, яка містить 'tables' або 'run_1'."
                )
                
    def open_results_viewer(self, target_dir=None):
        if target_dir: self.last_run_dir = target_dir

        if not self.last_run_dir or not os.path.exists(self.last_run_dir):
            messagebox.showerror("Помилка", "Результати не знайдено.")
            return

        self.setup_treeview_style()
        self.viewer = ctk.CTkToplevel(self)
        self.viewer.title(f"Детальні результати | {os.path.basename(self.last_run_dir)}")
        self.viewer.geometry("1150x800")
        self.viewer.grab_set() 

        runs = ["Чемпіон"]
        for d in sorted(os.listdir(self.last_run_dir)):
            if d.startswith("run_") and os.path.isdir(os.path.join(self.last_run_dir, d)):
                runs.append(f"Запуск {d.split('_')[1]}")

        top_panel = ctk.CTkFrame(self.viewer)
        top_panel.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(top_panel, text="Оберіть результат для перегляду:", font=self.font_main).pack(side="left", padx=10)
        
        self.run_selector = ctk.CTkOptionMenu(top_panel, values=runs, font=self.font_main, command=self.load_viewer_data)
        self.run_selector.set("Чемпіон")
        self.run_selector.pack(side="left", padx=10)

        tabs = ctk.CTkTabview(self.viewer)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_map = tabs.add("Карта та Рішення")
        self.tab_conv = tabs.add("Збіжність")
        self.tab_summary = tabs.add("Зведені дані")
        self.tab_inp = tabs.add("INP файл")
        self.tab_logs = tabs.add("Логи")

        self.paned_window = tk.PanedWindow(
            self.tab_map, 
            orient=tk.HORIZONTAL, 
            bg="#2b2b2b",            
            sashwidth=6,          
            sashrelief=tk.FLAT,        
            bd=0,                     
            cursor="sb_h_double_arrow" 
        )
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.map_canvas_frame = ctk.CTkFrame(self.paned_window)
        table_frame = ctk.CTkFrame(self.paned_window)

        self.paned_window.add(self.map_canvas_frame, stretch="always", minsize=400)
        self.paned_window.add(table_frame, stretch="never", minsize=500)
        
        self.viewer.update_idletasks()
        start_sash_pos = int(self.viewer.winfo_width() * 0.72)
        if start_sash_pos > 0:
            self.paned_window.paneconfigure(self.map_canvas_frame, width=start_sash_pos)
        
        scroll_y = ttk.Scrollbar(table_frame)
        scroll_y.pack(side="right", fill="y")
        self.tree_solution = ttk.Treeview(table_frame, yscrollcommand=scroll_y.set)
        scroll_y.config(command=self.tree_solution.yview)
        self.tree_solution.pack(fill="both", expand=True)

        self.conv_canvas_frame = ctk.CTkFrame(self.tab_conv)
        self.conv_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        summary_frame = ctk.CTkFrame(self.tab_summary)
        summary_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.tree_summary = ttk.Treeview(summary_frame)
        self.tree_summary.pack(fill="both", expand=True)

        self.txt_inp = ctk.CTkTextbox(self.tab_inp, font=self.font_mono, wrap="none")
        self.txt_inp.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_inp.bind("<Key>", self.prevent_typing) # Фікс тексту
        
        self.log_controls_frame = ctk.CTkFrame(self.tab_logs, fg_color="transparent")
        self.log_controls_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        ctk.CTkLabel(self.log_controls_frame, text="Оберіть лог-файл:", font=self.font_main).pack(side="left", padx=(0, 10))
        self.log_selector = ctk.CTkOptionMenu(self.log_controls_frame, values=["---"], font=self.font_main, command=self._on_log_select)
        self.log_selector.pack(side="left")
        
        self.viewer_txt_logs = ctk.CTkTextbox(self.tab_logs, font=self.font_mono, wrap="word")
        self.viewer_txt_logs.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        self.viewer_txt_logs.bind("<Key>", self.prevent_typing) # Фікс тексту
        
        self.available_logs = {}
        self.load_viewer_data("Чемпіон")

    def load_viewer_data(self, selected_run):
        for widget in self.map_canvas_frame.winfo_children(): widget.destroy()
        for widget in self.conv_canvas_frame.winfo_children(): widget.destroy()
        
        if selected_run == "Чемпіон":
            plots_dir = os.path.join(self.last_run_dir, "plots")
            tables_dir = os.path.join(self.last_run_dir, "tables")
            logs_dir = os.path.join(self.last_run_dir, "logs") 
            sol_name = "solution_champion.csv"
        else:
            run_id = selected_run.split(" ")[1]
            run_dir = os.path.join(self.last_run_dir, f"run_{run_id}")
            plots_dir = os.path.join(run_dir, "plots")
            tables_dir = os.path.join(run_dir, "tables")
            logs_dir = os.path.join(run_dir, "logs") 
            sol_name = "solution.csv"

        hist_name = "convergence_history.csv"
        sol_path = os.path.join(tables_dir, sol_name)
        
        actual_diams = self._get_actual_diams()

        diam_col_name = None
        if os.path.exists(sol_path):
            df_display = pd.read_csv(sol_path)
            diam_col_name = df_display.columns[-1] 
            for col in df_display.columns:
                if 'diam' in col.lower() or 'діаметр' in col.lower() or 'd_' in col.lower():
                    diam_col_name = col
                    break
                    
            if actual_diams:
                df_display[diam_col_name] = df_display[diam_col_name].apply(
                    lambda x: actual_diams[int(x)] if (pd.notnull(x) and float(x).is_integer() and 0 <= int(float(x)) < len(actual_diams)) else x
                )
            for col in df_display.columns:
                col_lower = str(col).lower()
                
                if 'id' in col_lower or 'node' in col_lower or 'вузол' in col_lower:
                    df_display[col] = df_display[col].apply(
                        lambda x: str(x)[:-2] if str(x).endswith('.0') else str(x)
                    )
                else:
                    def format_value(val):
                        if pd.isnull(val): return ""
                        try:
                            f_val = float(val)
                            if f_val.is_integer():
                                return str(int(f_val))
                            else:
                                return f"{f_val:.2f}"
                        except (ValueError, TypeError):
                            return str(val)
                            
                    df_display[col] = df_display[col].apply(format_value)

            self.load_df_to_treeview(self.tree_solution, df_display)

        fig_map, ax_map = plt.subplots(figsize=(6, 5), dpi=100)
        fig_map.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig_map.patch.set_facecolor('#ffffff')
        ax_map.set_facecolor('#ffffff')
        ax_map.set_aspect('equal', adjustable='datalim')
        
        target_inp = self.selected_inp
        saved_inp_path = os.path.join(tables_dir, "optimized_network.inp")
        
        if not target_inp or not os.path.exists(target_inp):
            if os.path.exists(saved_inp_path):
                target_inp = saved_inp_path

        drawn_native = False
        if os.path.exists(sol_path) and target_inp and os.path.exists(target_inp):
            try:
                geometry = netdraw.NetworkGeometry.load(target_inp)
                diameters = netdraw.diameters_from_table(
                    geometry, pd.read_csv(sol_path), actual_diams)

                scale = netdraw.draw(ax_map, geometry, diameters)
                scale.attach_colorbar(fig_map, ax_map,
                                      f"Діаметр ({self.opt_units.get()})")

                ax_map.set_title(f"Оптимізована топологія ({selected_run})", color="black", pad=10)
                
                canvas_map = FigureCanvasTkAgg(fig_map, master=self.map_canvas_frame)
                
                def on_resize(_event):
                    ax_map.autoscale_view(tight=True)
                    fig_map.canvas.draw_idle()
                fig_map.canvas.mpl_connect('resize_event', on_resize)
                
                canvas_map.get_tk_widget().pack(fill="both", expand=True)
                canvas_map.draw()
                self.enable_mouse_navigation(fig_map, ax_map, canvas_map)
                drawn_native = True
            except Exception as e:
                print(f"Native drawing failed: {e}")

        if not drawn_native:
            map_png = os.path.join(plots_dir, "network_map.png")
            if os.path.exists(map_png):
                img_map = Image.open(map_png)
                ctk_img = ctk.CTkImage(light_image=img_map, dark_image=img_map, size=(650, 600))
                ctk.CTkLabel(self.map_canvas_frame, image=ctk_img, text="").pack(expand=True)
            else:
                ctk.CTkLabel(self.map_canvas_frame, text="Дані карти не знайдено").pack(pady=20)
        plt.close(fig_map)

        fig_conv, ax_conv = plt.subplots(figsize=(8, 5), dpi=100)
        fig_conv.patch.set_facecolor('#2b2b2b')
        ax_conv.set_facecolor('#2b2b2b')
        ax_conv.tick_params(colors='white')
        ax_conv.xaxis.label.set_color('white')
        ax_conv.yaxis.label.set_color('white')
        ax_conv.set_xlabel("Симуляції")
        ax_conv.set_ylabel("Вартість (М$)")
        ax_conv.grid(True, color='gray', linestyle='--', alpha=0.5)

        hist_path = os.path.join(tables_dir, hist_name)
        if not os.path.exists(hist_path): 
            hist_path = os.path.join(plots_dir, hist_name) 
            
        if os.path.exists(hist_path):
            try:
                df_hist = pd.read_csv(hist_path)
                x_col = df_hist.columns[0]
                y_col = df_hist.columns[1]
                ax_conv.step(df_hist[x_col], df_hist[y_col] / 1e6, color='#00ffcc', linewidth=2, where='post')
                ax_conv.set_title(f"Збіжність: {selected_run}", color="white")
                
                canvas_conv = FigureCanvasTkAgg(fig_conv, master=self.conv_canvas_frame)
                canvas_conv.get_tk_widget().pack(side="top", fill="both", expand=True)
                canvas_conv.draw()
                self.enable_mouse_navigation(fig_conv, ax_conv, canvas_conv)
            except Exception as e: 
                ctk.CTkLabel(self.conv_canvas_frame, text=f"Помилка завантаження графіка: {e}").pack(pady=20)
        else:
            ctk.CTkLabel(self.conv_canvas_frame, text="Файл convergence_history.csv не знайдено").pack(pady=20)
        plt.close(fig_conv)

        sum_path = os.path.join(self.last_run_dir, "tables", "runs_summary.csv")
        if os.path.exists(sum_path):
            try:
                df_sum = pd.read_csv(sum_path)
                self.load_df_to_treeview(self.tree_summary, df_sum)
            except: pass

        inp_path = os.path.join(tables_dir, "optimized_network.inp")
        self.txt_inp.delete("1.0", "end")
        if os.path.exists(inp_path):
            try:
                with open(inp_path, "r", encoding="utf-8") as f: self.txt_inp.insert("end", f.read())
            except Exception as e: self.txt_inp.insert("end", f"Помилка: {e}")
        else:
            self.txt_inp.insert("end", "Файл INP не знайдено.")

        if hasattr(self, 'viewer_txt_logs'):
            self.viewer_txt_logs.delete("1.0", "end")
            
            target_logs_dir = None
            if os.path.exists(logs_dir): target_logs_dir = logs_dir
            elif os.path.exists(os.path.join(self.last_run_dir, "logs")):
                target_logs_dir = os.path.join(self.last_run_dir, "logs")
            
            self.available_logs.clear()
            display_names = []
            
            if target_logs_dir:
                log_files = [f for f in os.listdir(target_logs_dir) if f.endswith(".txt")]
                if log_files:
                    log_files.sort(key=lambda x: (not x.startswith('run_'), x))
                    
                    for lf in log_files:
                        path = os.path.join(target_logs_dir, lf)
                        if lf.startswith('run_'): disp_name = f"📄 Головний лог ({lf})"
                        else: disp_name = f"⚙️ {lf}"
                        self.available_logs[disp_name] = path
                        display_names.append(disp_name)
            
            if display_names:
                self.log_selector.configure(values=display_names)
                self.log_selector.set(display_names[0])
                self._on_log_select(display_names[0])
            else:
                self.log_selector.configure(values=["Немає логів"])
                self.log_selector.set("Немає логів")
                self.viewer_txt_logs.insert("end", "Файли логів (.txt) не знайдені.")
                
    def _on_log_select(self, selected_name):
        if not hasattr(self, 'viewer_txt_logs') or selected_name not in self.available_logs:
            return
        log_path = self.available_logs[selected_name]
        
        self.viewer_txt_logs.delete("1.0", "end")
        self.viewer_txt_logs.insert("end", f"Завантаження {selected_name}...\n")
        self.viewer_txt_logs.update() 
        try:
            self.viewer_txt_logs.delete("1.0", "end")
            with open(log_path, "r", encoding="utf-8") as f:
                self.viewer_txt_logs.insert("end", f.read())
        except Exception as e:
            self.viewer_txt_logs.insert("end", f"❌ Помилка читання файлу: {e}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = PipelineOptimizerApp()
    app.mainloop()