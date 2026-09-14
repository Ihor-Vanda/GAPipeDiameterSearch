"""Вікно детальних результатів прогону.

До цього модуля перегляд результатів жив трьома методами в
`PipelineOptimizerApp` (`open_results_viewer`, `load_viewer_data`,
`_on_log_select` — разом 289 рядків) і читав **тридцять** полів
застосунку. Аналіз показав, що шістнадцять із них створює він сам, а з
решти лише п'ять — справжні дані: каталог прогону, файл мережі, таблиця
цін, одиниці й шрифти. Решта дев'ять були спільними помічниками, які
переїхали в `gui_widgets`.

Тому тут звичайне вікно з п'ятьма аргументами, і головне вікно більше
нічого про його внутрішній устрій не знає.
"""

import os
import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import cost_table
import gui_widgets as widgets
import netdraw
import solution_table

#: Заголовок вкладки-зведення, коли переглядають найкращий прогін серії.
CHAMPION = "Чемпіон"
#: Файли, які вікно читає з каталогу прогону.
HISTORY_FILE = "convergence_history.csv"
SUMMARY_FILE = "runs_summary.csv"
OPTIMIZED_INP = "optimized_network.inp"
MAP_PNG = "network_map.png"


class ResultsViewer(ctk.CTkToplevel):
    """Окреме вікно: карта, рішення, збіжність, зведення, .inp і логи."""

    def __init__(self, parent, run_dir, inp_file, cost_file, units,
                 font_main, font_mono):
        super().__init__(parent)
        self.run_dir = run_dir
        self.inp_file = inp_file
        self.cost_file = cost_file
        self.units = units
        self.font_main = font_main
        self.font_mono = font_mono
        self.available_logs = {}

        self._build()

    def _build(self):
        widgets.setup_treeview_style(self)
        self.title(f"Детальні результати | {os.path.basename(self.run_dir)}")
        self.geometry("1150x800")
        self.grab_set() 

        runs = [CHAMPION]
        for d in sorted(os.listdir(self.run_dir)):
            if d.startswith("run_") and os.path.isdir(os.path.join(self.run_dir, d)):
                runs.append(f"Запуск {d.split('_')[1]}")

        top_panel = ctk.CTkFrame(self)
        top_panel.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(top_panel, text="Оберіть результат для перегляду:", font=self.font_main).pack(side="left", padx=10)
        
        self.run_selector = ctk.CTkOptionMenu(top_panel, values=runs, font=self.font_main, command=self.load_viewer_data)
        self.run_selector.set(CHAMPION)
        self.run_selector.pack(side="left", padx=10)

        tabs = ctk.CTkTabview(self)
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
        
        self.update_idletasks()
        start_sash_pos = int(self.winfo_width() * 0.72)
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
        self.txt_inp.bind("<Key>", widgets.prevent_typing) # Фікс тексту
        
        self.log_controls_frame = ctk.CTkFrame(self.tab_logs, fg_color="transparent")
        self.log_controls_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        ctk.CTkLabel(self.log_controls_frame, text="Оберіть лог-файл:", font=self.font_main).pack(side="left", padx=(0, 10))
        self.log_selector = ctk.CTkOptionMenu(self.log_controls_frame, values=["---"], font=self.font_main, command=self._on_log_select)
        self.log_selector.pack(side="left")
        
        self.viewer_txt_logs = ctk.CTkTextbox(self.tab_logs, font=self.font_mono, wrap="word")
        self.viewer_txt_logs.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        self.viewer_txt_logs.bind("<Key>", widgets.prevent_typing) # Фікс тексту
        
        self.load_viewer_data(CHAMPION)

    # --- звідки читати ------------------------------------------------

    def _run_paths(self, selected_run):
        """Каталоги і файл рішення для обраного прогону.

        Єдине, чим чемпіон відрізняється від окремого прогону, — корінь і
        назва файлу рішення. Раніше ця різниця розгорталась у вісім
        локальних змінних на початку 190-рядкового методу.
        """
        if selected_run == CHAMPION:
            base, sol_name = self.run_dir, "solution_champion.csv"
        else:
            run_id = selected_run.split(" ")[1]
            base = os.path.join(self.run_dir, f"run_{run_id}")
            sol_name = "solution.csv"

        tables = os.path.join(base, "tables")
        return {
            "plots": os.path.join(base, "plots"),
            "tables": tables,
            "logs": os.path.join(base, "logs"),
            "solution": os.path.join(tables, sol_name),
        }

    def load_viewer_data(self, selected_run):
        """Перечитати все, що показує вікно, для обраного прогону.

        Було одним методом на 190 рядків із цикломатичною складністю 44:
        шість незалежних читань ділили локальні змінні й порядок, тому
        зміна в одному зачіпала решту. Тепер кожне окремо (§27.5).
        """
        for widget in self.map_canvas_frame.winfo_children():
            widget.destroy()
        for widget in self.conv_canvas_frame.winfo_children():
            widget.destroy()

        paths = self._run_paths(selected_run)
        table = cost_table.diameter_table(self.cost_file)

        self._fill_solution_table(paths["solution"], table)
        self._fill_network_map(selected_run, paths, table)
        self._fill_convergence(selected_run, paths)
        self._fill_summary()
        self._fill_inp_text(paths["tables"])
        self._fill_log_list(paths["logs"])

    # --- шість незалежних читань --------------------------------------

    def _fill_solution_table(self, sol_path, table):
        """Таблиця рішення: індекси типорозмірів у діаметри, числа у вигляд."""
        if not os.path.exists(sol_path):
            return

        frame = solution_table.with_real_diameters(pd.read_csv(sol_path), table)
        widgets.load_dataframe(self.tree_solution, solution_table.for_display(frame))

    def _fill_network_map(self, selected_run, paths, table):
        """Карта мережі: власне малювання, а якщо немає з чого — готовий PNG."""
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal", adjustable="datalim")

        if not self._draw_native_map(fig, ax, selected_run, paths, table):
            self._show_map_png(paths["plots"])
        plt.close(fig)

    def _draw_native_map(self, fig, ax, selected_run, paths, table):
        """Карта з `.inp` і `solution.csv`; `False` — нема з чого малювати."""
        target_inp = self.inp_file
        saved_inp = os.path.join(paths["tables"], OPTIMIZED_INP)
        if not target_inp or not os.path.exists(target_inp):
            target_inp = saved_inp if os.path.exists(saved_inp) else target_inp

        if not (os.path.exists(paths["solution"])
                and target_inp and os.path.exists(target_inp)):
            return False

        try:
            geometry = netdraw.NetworkGeometry.load(target_inp)
            diameters = netdraw.diameters_from_table(
                geometry, pd.read_csv(paths["solution"]), table)

            scale = netdraw.draw(ax, geometry, diameters)
            scale.attach_colorbar(fig, ax, f"Діаметр ({self.units})")
            ax.set_title(f"Оптимізована топологія ({selected_run})",
                         color="black", pad=10)

            canvas = FigureCanvasTkAgg(fig, master=self.map_canvas_frame)

            def on_resize(_event):
                ax.autoscale_view(tight=True)
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("resize_event", on_resize)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()
            widgets.enable_mouse_navigation(fig, ax, canvas)
            return True
        except Exception as error:
            print(f"Native drawing failed: {error}")
            return False

    def _show_map_png(self, plots_dir):
        """Запасний шлях: картинка зі звіту, якщо мережу відновити не вдалось."""
        map_png = os.path.join(plots_dir, MAP_PNG)
        if not os.path.exists(map_png):
            ctk.CTkLabel(self.map_canvas_frame,
                         text="Дані карти не знайдено").pack(pady=20)
            return

        image = Image.open(map_png)
        ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=(650, 600))
        ctk.CTkLabel(self.map_canvas_frame, image=ctk_img, text="").pack(expand=True)

    def _fill_convergence(self, selected_run, paths):
        """Графік збіжності; історія може лежати і в `tables`, і в `plots`."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor("#2b2b2b")
        ax.set_facecolor("#2b2b2b")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.set_xlabel("Симуляції")
        ax.set_ylabel("Вартість (М$)")
        ax.grid(True, color="gray", linestyle="--", alpha=0.5)

        hist_path = os.path.join(paths["tables"], HISTORY_FILE)
        if not os.path.exists(hist_path):
            hist_path = os.path.join(paths["plots"], HISTORY_FILE)

        if not os.path.exists(hist_path):
            ctk.CTkLabel(self.conv_canvas_frame,
                         text=f"Файл {HISTORY_FILE} не знайдено").pack(pady=20)
        else:
            try:
                frame = pd.read_csv(hist_path)
                sims, cost = frame.columns[0], frame.columns[1]
                ax.step(frame[sims], frame[cost] / 1e6, color="#00ffcc",
                        linewidth=2, where="post")
                ax.set_title(f"Збіжність: {selected_run}", color="white")

                canvas = FigureCanvasTkAgg(fig, master=self.conv_canvas_frame)
                canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
                canvas.draw()
                widgets.enable_mouse_navigation(fig, ax, canvas)
            except Exception as error:
                ctk.CTkLabel(self.conv_canvas_frame,
                             text=f"Помилка завантаження графіка: {error}").pack(pady=20)
        plt.close(fig)

    def _fill_summary(self):
        """Зведення серії — одне на всі прогони, тому читається з кореня."""
        sum_path = os.path.join(self.run_dir, "tables", SUMMARY_FILE)
        if not os.path.exists(sum_path):
            return
        try:
            widgets.load_dataframe(self.tree_summary, pd.read_csv(sum_path))
        except Exception:
            pass

    def _fill_inp_text(self, tables_dir):
        """Текст оптимізованої мережі."""
        inp_path = os.path.join(tables_dir, OPTIMIZED_INP)
        self.txt_inp.delete("1.0", "end")

        if not os.path.exists(inp_path):
            self.txt_inp.insert("end", "Файл INP не знайдено.")
            return
        try:
            with open(inp_path, "r", encoding="utf-8") as handle:
                self.txt_inp.insert("end", handle.read())
        except Exception as error:
            self.txt_inp.insert("end", f"Помилка: {error}")

    def _fill_log_list(self, logs_dir):
        """Список логів: головний лог прогону першим, воркери за назвою."""
        self.viewer_txt_logs.delete("1.0", "end")
        self.available_logs.clear()

        target = logs_dir
        if not os.path.exists(target):
            target = os.path.join(self.run_dir, "logs")

        names = []
        if os.path.exists(target):
            files = sorted((f for f in os.listdir(target) if f.endswith(".txt")),
                           key=lambda name: (not name.startswith("run_"), name))
            for name in files:
                label = (f"📄 Головний лог ({name})" if name.startswith("run_")
                         else f"⚙️ {name}")
                self.available_logs[label] = os.path.join(target, name)
                names.append(label)

        if names:
            self.log_selector.configure(values=names)
            self.log_selector.set(names[0])
            self._on_log_select(names[0])
        else:
            self.log_selector.configure(values=["Немає логів"])
            self.log_selector.set("Немає логів")
            self.viewer_txt_logs.insert("end", "Файли логів (.txt) не знайдені.")

    def _on_log_select(self, selected_name):
        if selected_name not in self.available_logs:
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
