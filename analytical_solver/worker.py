"""Єдина реалізація запуску острова-воркера.

До цього модуля запуск воркера існував у трьох місцях:

* `AnalyticalSolver.worker_task` (classmethod) — перезаписувався на старті;
* `analytical_worker_task` у `main.py` — 82 рядки;
* `analytical_worker_task` у `gui.py` — ті самі 82 рядки, побайтово.

Обидві точки входу робили мавпячий патч
`AnalyticalSolver.worker_task = staticmethod(analytical_worker_task)`, а
класи солвера діставали через `sys.modules[AnalyticalSolver.__module__]` —
тобто через приватний простір імен чужого модуля. Це працювало лише тому,
що `orchestrator.py` випадково мав потрібні імена, і будь-яка зміна
імпортів у пакеті зламала б GUI без помилки на етапі імпорту.

Тепер запуск один, симулятор процесу належить цьому модулю, а не
`__main__`, і передається явно.
"""

import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .context import SolverContext, beam_width_for, classify_network
from .kicks import KickStrategies
from .local_search import LocalSearch

#: Симулятор поточного процесу. У паралельному режимі його створює
#: `init_worker_process` (initializer пулу), у послідовному —
#: `set_simulator` у тому ж процесі. Раніше цю роль грав атрибут
#: `worker_sim_instance` у `__main__`, який кожна точка входу
#: оголошувала сама.
_simulator = None


def get_temp_root():
    """Корінь тимчасових каталогів воркерів (окремий каталог на процес)."""
    return os.path.abspath("_temp_sim_data")


def clean_temp_root():
    """Прибрати всі тимчасові каталоги воркерів.

    Версія з `gui.py`: на Windows EPANET Toolkit тримає дескриптори
    .rpt/.bin навіть після завершення воркера, тому `rmtree` іноді падає
    з першого разу — звідси спроби з паузою. У `main.py` була спрощена
    версія на 9 рядків, яка на Windows лишала сміття.
    """
    import time

    root = get_temp_root()

    # Якщо поточний каталог усередині кореня, вийти з нього, інакше
    # Windows не дасть його видалити.
    try:
        if os.path.abspath(os.getcwd()).startswith(os.path.abspath(root)):
            os.chdir(os.path.dirname(os.path.abspath(root)))
    except OSError:
        pass

    for name in os.listdir(os.getcwd()):
        low = name.lower()
        if low.startswith('temp') and low.endswith(('.bin', '.inp', '.rpt', '.out')):
            try:
                os.remove(os.path.join(os.getcwd(), name))
            except OSError:
                pass

    if not os.path.exists(root):
        return

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(('.bin', '.inp', '.rpt', '.out')):
                try:
                    os.remove(os.path.join(dirpath, name))
                except OSError:
                    pass

    for _ in range(10):
        try:
            shutil.rmtree(root, ignore_errors=False)
            return
        except OSError:
            time.sleep(0.5)
    shutil.rmtree(root, ignore_errors=True)


def set_simulator(simulator):
    """Задати симулятор для запуску воркера в поточному процесі."""
    global _simulator
    _simulator = simulator


def init_worker_process(inp_file, config):
    """Initializer процесу пулу: свій каталог, своя копія .inp, свій симулятор.

    Кожен воркер працює у власному каталозі, бо EPANET Toolkit пише
    тимчасові .rpt/.bin файли поруч і воркери інакше б їх перетирали.
    """
    import signal

    # Ctrl+C обробляє головний процес; воркери мусять його ігнорувати,
    # інакше KeyboardInterrupt рве пул замість м'якої зупинки.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from water_sim import WaterSimulator  # локальний імпорт: пакет солвера
                                              # не має залежати від EPANET-адаптера

        pid = os.getpid()
        temp_dir = os.path.join(get_temp_root(), f"worker_{pid}")
        os.makedirs(temp_dir, exist_ok=True)
        os.chdir(temp_dir)

        base_name = os.path.splitext(os.path.basename(inp_file))[0]
        local_inp = os.path.abspath(f"{base_name}_worker_{pid}.inp")
        shutil.copy2(inp_file, local_inp)

        set_simulator(WaterSimulator(local_inp, config, temp_dir=temp_dir))
    except Exception as e:
        sys.stderr.write(f"!!! WORKER {os.getpid()} INIT CRASH: {e}\n")
        sys.exit(1)


@dataclass(frozen=True)
class WorkerTask:
    """Завдання одного острова.

    Замінює позиційний кортеж на 12 елементів, який розпаковувався
    вручну в трьох місцях, з двома `if len(args) > N` на хвості.
    """

    diameters: list
    v_opt: float
    time_budget: float
    global_best_cost: float
    seed: int
    worker_id: int
    n_workers: int = 1
    max_sims: float = float("inf")
    shared_progress: object = None
    log_dir: str | None = None
    # `global_archive` при створенні завдання завжди порожній: реальне
    # значення воркер отримує з shared_progress під час міграції (§2.3).
    global_archive: list = field(default_factory=list)
    # Епохи — історичний артефакт, epoch завжди 0 (§2.3). Поле лишається
    # доки епохи не прибрані повністю, бо від нього залежить вибір seeds.
    epoch: int = 0


class WorkerResult(NamedTuple):
    """Результат острова. Раніше — кортеж із порожнім третім елементом."""

    best_cost: float
    best_solution: list | None
    sim_count: int
    basin_tabu: object


class _WorkerLogger:
    """Перенаправляє stdout воркера у власний файл."""

    def __init__(self, handle):
        self.handle = handle

    def write(self, message):
        self.handle.write(message)
        self.handle.flush()

    def writelines(self, lines):
        self.handle.writelines(lines)
        self.handle.flush()

    def flush(self):
        self.handle.flush()


def _open_worker_log(task):
    """Відкрити лог воркера, якщо задано каталог. Повертає handle або None."""
    if not task.log_dir:
        return None

    logs_folder = task.log_dir if "logs" in task.log_dir else os.path.join(task.log_dir, "logs")
    os.makedirs(logs_folder, exist_ok=True)
    handle = open(os.path.join(logs_folder, f"worker_{task.worker_id + 1:02d}.txt"),
                  "a", encoding="utf-8")
    handle.write(f"\n\n{'=' * 50}\n 🚀 WORKER {task.worker_id + 1:02d}\n{'=' * 50}\n")
    handle.flush()
    return handle


def run_island_worker(task):
    """Запустити один острів і повернути його найкраще рішення.

    Викликається і як `pool.apply_async(run_island_worker, (task,))`, і
    напряму в послідовному режимі — різниця лише в тому, хто до цього
    викликав `init_worker_process` / `set_simulator`.
    """
    if _simulator is None:
        raise RuntimeError(
            "Симулятор воркера не ініціалізований: викличте init_worker_process() "
            "як initializer пулу або set_simulator() у поточному процесі."
        )

    random.seed(task.seed)
    np.random.seed(task.seed)

    original_stdout = sys.stdout
    handle = _open_worker_log(task)

    try:
        if handle:
            sys.stdout = _WorkerLogger(handle)

        # Локальний імпорт: IslandWorker живе в orchestrator.py, який
        # імпортує цей модуль. Знімається, коли IslandWorker переїде у
        # власний файл (§5 плану).
        from .orchestrator import IslandWorker

        ctx = SolverContext(_simulator, task.diameters, v_opt=task.v_opt)
        local_search = LocalSearch(ctx)
        kicker = KickStrategies(ctx, local_search)

        network_class = classify_network(ctx.num_pipes)
        worker = IslandWorker(
            ctx, kicker, local_search, task.worker_id, task.n_workers,
            task.max_sims, beam_width_for(network_class), network_class,
            task.global_archive, task.epoch,
        )
        best_cost, best_solution = worker.run(task.time_budget, task.global_best_cost,
                                              task.shared_progress)
        return WorkerResult(best_cost, best_solution, ctx.sim_count, worker.pool.basin_tabu)

    except Exception as e:
        print(f"     [CRITICAL WORKER ERROR] Worker {task.worker_id + 1} failed: {e}")
        return WorkerResult(float("inf"), None, 0, set())

    finally:
        sys.stdout = original_stdout
        if handle:
            handle.close()
