"""Сценарій запуску оптимізації, спільний для CLI та GUI.

До цього модуля той самий сценарій існував двічі: `main()` у `main.py`
(154 рядки) і `PipelineOptimizerApp.run_optimization()` у `gui.py`
(171 рядок). Обидві реалізації робили одне й те саме — збирали `SolverConfig`,
підіймали `WaterSimulator`, створювали пул, крутили цикл `runs`,
перетворювали діаметри назад в індекси, вибирали чемпіона й експортували
звіти, — і вже розійшлися в деталях (CLI при `runs > 1` зберігав рішення
без історії та без графіків, GUI — з ними).

Тепер сценарій один, а від точки входу лишається тільки те, що справді
різне: звідки беруться параметри і куди йде прогрес.
"""

import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from analytical_solver import AnalyticalSolver
from analytical_solver.worker import get_temp_root, init_worker_process, set_simulator
from config import SolverConfig
from console import format_time
from cost_table import load_cost_table
from plot import export_solution, plot_convergence, plot_network_map
from water_sim import WaterSimulator

RUN_MODES = ("analytical", "fast_analytical")


def _to_int(text, default=0):
    """Толерантний розбір числа з текстового поля GUI."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return default


def _to_float(text, default=0.0):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


@dataclass
class RunRequest:
    """Усе, що потрібно знати, щоб провести серію прогонів."""

    inp_file: str
    cost_file: str
    h_min: float = 30.0
    units: str = "mm"
    v_opt: float = 1.0
    runs: int = 1
    #: 0 = усі доступні ядра
    cores: int = 0
    #: None = бюджет за класом мережі
    max_sims: int | None = None
    run_mode: str = "analytical"
    #: None = вивести з калібрування; 0 або менше = без часового обмеження
    time_limit: float | None = None
    #: Один острів у цьому ж процесі — єдиний відтворюваний режим (§12.4)
    sequential: bool = False

    def __post_init__(self):
        if self.run_mode not in RUN_MODES:
            raise ValueError(f"run_mode={self.run_mode!r}; очікується одне з {RUN_MODES}")
        if self.runs < 1:
            raise ValueError("runs мусить бути >= 1")

    @classmethod
    def from_text_fields(cls, *, inp_file, cost_file, hmin, units, v_opt, runs,
                         cores, max_sims, run_mode, **kwargs):
        """Зібрати запит із текстових полів GUI.

        Поля вводу можуть містити що завгодно, тому розбір толерантний —
        рівно так, як це робив `run_optimization` (`try: int(...) except: 0`).
        Виділено окремо, щоб цю логіку можна було перевірити тестами без
        запуску tkinter.
        """
        return cls(
            inp_file=inp_file,
            cost_file=cost_file,
            h_min=_to_float(hmin, 30.0),
            units=units,
            v_opt=_to_float(v_opt, 1.0),
            runs=max(1, _to_int(runs, 1)),
            cores=_to_int(cores, 0),
            max_sims=_to_int(max_sims, 0) or None,
            run_mode=run_mode,
            **kwargs,
        )

    @property
    def effective_cores(self):
        if self.sequential:
            return 1
        return self.cores if self.cores > 0 else multiprocessing.cpu_count()

    @property
    def solver_time_limit(self):
        """None — вивести з калібрування; inf — зупинка лише за бюджетом симуляцій."""
        limit = self.time_limit
        if limit is None and self.sequential:
            # Часовий бюджет за замовчуванням залежить від виміряної швидкості
            # симулятора, тому точка зупинки залежить від завантаження машини.
            # Для відтворюваного прогону зупиняємось лише за симуляціями.
            return float("inf")
        if limit is None:
            return None
        return float("inf") if limit <= 0 else limit


@dataclass
class RunResult:
    """Результат одного прогону."""

    run_id: int
    cost: float
    pressure: float
    feasible: bool
    duration: float
    individual: list
    history: list = field(default_factory=list)

    @property
    def total_evals(self):
        if not self.history:
            return 0
        return self.history[-1].get("evals", len(self.history))

    @property
    def status(self):
        return "OK" if self.feasible else "FAIL"

    def as_summary_row(self):
        return {"Run": self.run_id, "Cost": self.cost, "Pressure": self.pressure,
                "Feasible": self.feasible, "Time": self.duration}


def setup_run_directory(base_dir="OutputDataExperiments"):
    run_dir = os.path.join(base_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class OptimizationRunner:
    """Проводить серію прогонів і зберігає звіти. Не знає ні про CLI, ні про GUI.

    `on_progress(sims, best_cost, best_solution)` — необов'язковий колбек
    живого прогресу; GUI передає туди оновлення дашборда.
    `should_stop()` — необов'язкова перевірка між прогонами; GUI передає
    туди стан кнопки «ЗУПИНИТИ».
    `on_run_start(run_id)` — необов'язковий хук перед кожним прогоном
    (GUI очищає графік збіжності).
    """

    def __init__(self, request, on_progress=None, should_stop=None, on_run_start=None):
        self.request = request
        self.on_progress = on_progress
        self.should_stop = should_stop or (lambda: False)
        self.on_run_start = on_run_start

        self.config = None
        self.simulator = None
        self.pool = None

    # --- підготовка -------------------------------------------------------

    def prepare(self):
        """Зібрати конфіг, підняти симулятор і перевірити його на працездатність."""
        req = self.request
        self.config = SolverConfig(
            cost_file=req.cost_file, h_min=req.h_min, unit_system=req.units,
            run_mode=req.run_mode, v_opt=req.v_opt,
        )
        load_cost_table(self.config)

        temp_dir = os.path.abspath(os.path.join(get_temp_root(), "main_process"))
        os.makedirs(temp_dir, exist_ok=True)
        self.simulator = WaterSimulator(req.inp_file, self.config, temp_dir=temp_dir)
        # Pre-flight: один розрахунок, щоб упасти тут, а не в воркері.
        # Раніше тут стояв `evaluate`, який на збої симуляції повертав
        # 1e15, а результат відкидався — тобто перевірка не могла впасти
        # в принципі (§30.4).
        probe = self.simulator.get_stats([0] * self.simulator.n_variables)
        if not probe.simulation_ok:
            raise RuntimeError(
                f"Гідравлічний розрахунок не зійшовся на {req.inp_file}: "
                f"перевірте одиниці (--units {req.units}) і файл мережі")
        return self.config

    def _start_workers(self):
        req = self.request
        if req.sequential:
            set_simulator(self.simulator)
            print("   [Sequential] Pool вимкнено — прогін детермінований\n")
            return
        self.pool = multiprocessing.Pool(
            processes=req.effective_cores,
            initializer=init_worker_process,
            initargs=(req.inp_file, self.config),
        )

    def _stop_workers(self):
        if self.pool is not None:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception:
                pass
            self.pool = None

    # --- прогони ----------------------------------------------------------

    def _indices_of(self, diameters_m):
        """Зворотне відображення діаметрів у індекси таблиці цін."""
        table = self.config.diameters_m
        indices = []
        for d in diameters_m:
            try:
                indices.append(table.index(d))
            except ValueError:
                indices.append(len(table) - 1)
        return indices

    def _single_run(self, run_id, log_dir):
        req = self.request
        solver = AnalyticalSolver(
            self.simulator, self.config.diameters_m, v_opt=req.v_opt, pool=self.pool,
            log_dir=log_dir, n_workers=req.effective_cores, max_sims=req.max_sims,
            time_limit_sec=req.solver_time_limit,
        )

        started = time.time()
        if req.run_mode == "fast_analytical":
            diameters = solver.solve_fast(ui_callback=self.on_progress)
        else:
            diameters = solver.solve_standalone(ui_callback=self.on_progress)
        duration = time.time() - started

        if not diameters:
            return None

        individual = self._indices_of(diameters)
        stats = self.simulator.get_stats(individual)

        raw_history = getattr(solver, "history", None) or [(solver.ctx.sim_count, stats.cost)]
        history = [{"evals": sims, "min_cost": cost} for sims, cost in raw_history]

        return RunResult(
            run_id=run_id, cost=stats.cost, pressure=stats.p_min,
            feasible=stats.is_feasible(req.h_min), duration=duration,
            individual=individual, history=history,
        )

    def run(self, base_dir):
        """Провести всі прогони. Повертає список результатів (може бути порожнім)."""
        req = self.request
        results = []
        try:
            self._start_workers()

            for i in range(req.runs):
                if self.should_stop():
                    break
                if self.on_run_start:
                    self.on_run_start(i + 1)

                log_dir = os.path.abspath(base_dir if req.runs == 1
                                          else os.path.join(base_dir, f"run_{i + 1}"))
                os.makedirs(log_dir, exist_ok=True)

                try:
                    result = self._single_run(i + 1, log_dir)
                except KeyboardInterrupt:
                    # GUI зупиняє солвер саме так — через ui_callback.
                    print("\n[System] 🛑 Прогін перервано.")
                    break
                except Exception as e:
                    if self.should_stop():
                        break
                    print(f"\n[Error] Прогін {i + 1} завершився помилкою: {e}")
                    traceback.print_exc()
                    continue

                if result is None:
                    continue

                results.append(result)
                print(f"\nRun {result.run_id}: {result.cost/1e6:.4f}M$ | "
                      f"P={result.pressure:.2f}m | {result.status} | "
                      f"Time: {format_time(result.duration)}")

                if req.runs > 1:
                    self._export_one(result, log_dir)

        except KeyboardInterrupt:
            print("\n[System] 🛑 Перервано з клавіатури (Ctrl+C).")
        finally:
            self._stop_workers()

        return results

    # --- звіти ------------------------------------------------------------

    def _export_one(self, result, run_dir):
        tables = os.path.join(run_dir, "tables")
        plots = os.path.join(run_dir, "plots")
        os.makedirs(tables, exist_ok=True)
        os.makedirs(plots, exist_ok=True)

        export_solution(result.individual, result.history, self.request.inp_file,
                        os.path.join(tables, "solution"), self.config,
                        result.cost, result.duration, result.total_evals)
        plot_network_map(result.individual, self.request.inp_file,
                         os.path.join(plots, "network_map.png"), self.config, result.cost)
        if len(result.history) > 1:
            plot_convergence(result.history, os.path.join(plots, "convergence.png"))

    @staticmethod
    def champion(results):
        """Найкращий прогін: спершу допустимі, потім за вартістю."""
        if not results:
            return None
        return sorted(results, key=lambda r: (not r.feasible, r.cost))[0]

    def export(self, results, base_dir):
        """Зберегти зведення й звіти чемпіона. Повертає чемпіона або None."""
        if not results:
            return None

        tables = os.path.join(base_dir, "tables")
        plots = os.path.join(base_dir, "plots")
        os.makedirs(tables, exist_ok=True)
        os.makedirs(plots, exist_ok=True)

        print("\n=== FINAL RESULTS ===")
        for r in results:
            print(f"Run {r.run_id}: {r.cost/1e6:.4f}M$ | P={r.pressure:.2f}m | {r.status}")

        best = self.champion(results)
        export_solution(
            individual=best.individual, history=best.history,
            inp_file=self.request.inp_file,
            filename_prefix=os.path.join(tables, "solution_champion"),
            config=self.config, cost=best.cost, time_sec=best.duration,
            total_sims=best.total_evals,
        )
        plot_network_map(best.individual, self.request.inp_file,
                         os.path.join(plots, "network_map.png"), self.config, best.cost)
        if len(best.history) > 1:
            plot_convergence(best.history, os.path.join(plots, "convergence.png"))

        pd.DataFrame([r.as_summary_row() for r in results]).to_csv(
            os.path.join(tables, "runs_summary.csv"), index=False)

        return best
