import warnings

warnings.filterwarnings("ignore", message="Ignoring fixed .* limits")

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import threading
import sys
import os
import time
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ga_data
import netdraw
import gui_widgets as widgets
from analytical_solver.worker import clean_temp_root, get_temp_root
from gui_results import ResultsViewer
from gui_widgets import GUIStream
from runner import OptimizationRunner, RunRequest

# ==========================================
# 1 КЛАС GUI ТА ПЕРЕНАПРАВЛЕННЯ ЛОГІВ
# ==========================================

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


        widgets.setup_treeview_style(self) 
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
        self.txt_logs.bind("<Key>", widgets.prevent_typing)

        self.progress = ctk.CTkProgressBar(self.log_frame)
        self.progress.grid(row=2, column=0, pady=10, sticky="ew")
        self.progress.set(0)
        

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
                widgets.load_dataframe(self.tree_costs_preview, df)
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
                self._live_geometry, best_sol, ga_data.diameter_table(self.selected_costs))

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
            widgets.enable_mouse_navigation(self.topo_fig, self.topo_ax, self.topo_canvas) 
            
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg)
            ctk.CTkLabel(self.tab_topology, text=f"Не вдалося побудувати граф:\n{e}", font=self.font_main).pack(pady=20)
    
    # ==========================================
    # 7 Results Viewer
    # ==========================================

    def open_results_viewer(self, target_dir=None):
        """Відкрити вікно детальних результатів.

        Раніше цей метод сам будував вікно (88 рядків) і разом із
        `load_viewer_data` та `_on_log_select` читав 30 полів застосунку.
        Тепер вікно — окремий клас, який отримує п'ять значень (§27).
        """
        if target_dir:
            self.last_run_dir = target_dir

        if not self.last_run_dir or not os.path.exists(self.last_run_dir):
            messagebox.showerror("Помилка", "Результати не знайдено.")
            return

        ResultsViewer(self, run_dir=self.last_run_dir, inp_file=self.selected_inp,
                      cost_file=self.selected_costs, units=self.opt_units.get(),
                      font_main=self.font_main, font_mono=self.font_mono)

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
                

                

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = PipelineOptimizerApp()
    app.mainloop()