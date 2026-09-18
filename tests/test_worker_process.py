"""Підготовка процесу воркера: власний каталог, власна копія `.inp`, лог.

`init_worker_process` — це initializer пулу. Він виконується **в чужому
процесі**, до першої симуляції, і саме тому лишався непокритим: побачити
його помилку можна було лише як `!!! WORKER … INIT CRASH` у консолі або
як мовчазну відсутність результату.

Ціна помилки тут висока й непропорційна розміру функції. EPANET Toolkit
пише тимчасові `.rpt`/`.bin` **поруч із поточним каталогом** процесу.
Якщо воркери не розійдуться по власних каталогах, вони перетирають файли
одне одного, і пошук дає числа, яких ніхто не рахував (§29.1, §31.x —
половина дефектів прибирання походить звідси).

Тут перевіряється те, що робить цю ізоляцію: каталог на процес, копія
`.inp` у ньому, симулятор, прив'язаний саме до цієї копії.
"""

import os
import signal

import pytest

from pipeopt.solver import worker as worker_module
from pipeopt.solver.worker import (
    TEMP_ROOT_NAME, get_temp_root, init_worker_process, parent_of_temp_root,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANOI = os.path.join(ROOT, "InputData", "Hanoi", "Hanoi.inp")
HANOI_COSTS = os.path.join(ROOT, "InputData", "Hanoi", "costs.csv")


# --- вихід із тимчасового кореня ----------------------------------------
#
# Дві межові гілки залежать від ОС, і саме тому вони винесені в чисту
# функцію: корінь файлової системи і корінь диска Windows. Перевірити їх
# «як є» означало б мати права на `/` і працювати під Windows одночасно.

def test_a_path_outside_the_temp_root_means_nothing_to_do():
    assert parent_of_temp_root("/home/user/project", sep="/") is None


def test_the_parent_of_the_temp_root_is_the_project_directory():
    assert parent_of_temp_root(
        f"/home/user/project/{TEMP_ROOT_NAME}/worker_101", sep="/") \
        == "/home/user/project"


def test_the_temp_root_at_the_filesystem_root_leaves_us_at_the_root():
    """`/_temp_sim_data` — відкидання сегмента дає порожній рядок, не шлях."""
    assert parent_of_temp_root(f"/{TEMP_ROOT_NAME}/worker_1", sep="/") == "/"


def test_the_temp_root_on_a_windows_drive_keeps_the_backslash():
    """`C:` — це не каталог, а «поточний каталог диска C».

    `os.chdir("C:")` на Windows переходить не в корінь диска, а туди, де
    процес востаннє був на цьому диску, — тобто, можливо, назад у
    тимчасовий корінь. Тому потрібен саме `C:\\`.
    """
    assert parent_of_temp_root(f"C:\\{TEMP_ROOT_NAME}\\worker_1", sep="\\") \
        == "C:\\"


def test_a_nested_project_path_on_windows_works_too():
    assert parent_of_temp_root(
        f"E:\\Projects\\Python\\App\\{TEMP_ROOT_NAME}\\worker_7", sep="\\") \
        == "E:\\Projects\\Python\\App"


def test_only_the_first_occurrence_counts():
    """Каталог користувача теж може зватись `_temp_sim_data`.

    Вихід має бути **над найвищим** входженням: усе нижче лежить
    усередині кореня, який ми прибираємо.
    """
    assert parent_of_temp_root(
        f"/home/{TEMP_ROOT_NAME}/worker_1/{TEMP_ROOT_NAME}", sep="/") == "/home"


# --- initializer процесу пулу -------------------------------------------

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Ізоляція трьох глобальних речей, які чіпає initializer.

    Він змінює поточний каталог процесу, ставить глобальний симулятор і
    **перехоплює SIGINT**. Останнє найнеприємніше: без відновлення тест
    відібрав би Ctrl+C у самого `pytest`.
    """
    monkeypatch.chdir(tmp_path)
    previous_handler = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(worker_module, "_simulator", None, raising=False)
    yield tmp_path
    signal.signal(signal.SIGINT, previous_handler)


@pytest.fixture
def config():
    from pipeopt.config import SolverConfig
    from pipeopt.reporting.cost_table import load_cost_table

    cfg = SolverConfig(cost_file=HANOI_COSTS, unit_system="in", h_min=30.0)
    load_cost_table(cfg, silent=True)
    return cfg


pytestmark = pytest.mark.skipif(not os.path.exists(HANOI),
                                reason="немає InputData/Hanoi")


def test_the_worker_gets_a_directory_of_its_own(isolated, config):
    """Каталог іменується за PID — саме це розводить воркери між собою.

    Очікуваний шлях рахується **до** виклику: `get_temp_root()` бере
    корінь відносно поточного каталогу, а initializer цей каталог
    змінює. Та сама пастка, через яку прибирання тихо не працювало
    (§31.2).
    """
    expected = os.path.join(get_temp_root(), f"worker_{os.getpid()}")

    init_worker_process(HANOI, config)

    assert os.path.isdir(expected)
    assert os.path.samefile(os.getcwd(), expected)


def test_the_input_file_is_copied_next_to_the_worker(isolated, config):
    """EPANET пише `.rpt`/`.bin` поруч із `.inp`; спільний `.inp` = спільні файли."""
    init_worker_process(HANOI, config)

    copies = [name for name in os.listdir(os.getcwd()) if name.endswith(".inp")]
    assert copies, "воркер працює з чужим .inp"
    assert str(os.getpid()) in copies[0], "копія не іменована за процесом"


def test_the_copy_is_identical_to_the_original(isolated, config):
    init_worker_process(HANOI, config)

    copy = next(name for name in os.listdir(os.getcwd()) if name.endswith(".inp"))
    with open(copy, "rb") as made, open(HANOI, "rb") as source:
        assert made.read() == source.read()


def test_the_simulator_is_ready_after_initialisation(isolated, config):
    """Після initializer'а воркер має право одразу рахувати."""
    init_worker_process(HANOI, config)

    simulator = worker_module._simulator
    assert simulator is not None
    assert simulator.n_variables > 0
    try:
        assert simulator.get_stats([0] * simulator.n_variables).simulation_ok
    finally:
        simulator.close()


def test_interrupts_are_left_to_the_main_process(isolated, config):
    """Ctrl+C у воркері рве пул замість м'якої зупинки.

    Тому воркер ігнорує SIGINT: сигнал обробляє головний процес, а пул
    завершується штатно.
    """
    init_worker_process(HANOI, config)

    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    if worker_module._simulator is not None:
        worker_module._simulator.close()


def test_a_broken_input_file_stops_the_worker_loudly(isolated, config, capsys):
    """Мовчазний воркер без симулятора дав би `None` на кожній симуляції.

    `sys.exit(1)` у initializer'і — єдиний спосіб повідомити пул, що цей
    процес непридатний; повідомлення йде в `stderr`, бо `stdout` воркера
    може бути перенаправлений у файл логу.
    """
    with pytest.raises(SystemExit) as stopped:
        init_worker_process(os.path.join(str(isolated), "no-such-file.inp"), config)

    assert stopped.value.code == 1
    assert "INIT CRASH" in capsys.readouterr().err


# --- лог воркера --------------------------------------------------------

def test_every_line_written_by_a_worker_reaches_the_file_at_once(tmp_path):
    """Лог воркера читають, поки прогін ще йде — буферизований лог марний.

    Прогін триває годинами; якщо рядки лежать у буфері до закриття файлу,
    подивитись «що зараз робить острів» неможливо, а після аварійного
    завершення хвіст логу втрачається взагалі.
    """
    from pipeopt.solver.worker import _WorkerLogger

    path = tmp_path / "worker.log"
    with open(path, "w", encoding="utf-8") as handle:
        writer = _WorkerLogger(handle)

        writer.write("рядок один\n")
        assert path.read_text(encoding="utf-8") == "рядок один\n"

        writer.writelines(["два\n", "три\n"])
        assert path.read_text(encoding="utf-8").endswith("два\nтри\n")

        writer.flush()


# --- аварія в самому острові ---------------------------------------------

def test_an_uninitialised_worker_refuses_loudly():
    """Без симулятора воркер не має що рахувати — і каже це винятком.

    Виняток тут навмисно **не** ловиться: він летить у пул, а
    оркестратор друкує `[Error] Worker N crashed`. Мовчазне повернення
    порожнього результату виглядало б як «острів нічого не знайшов»,
    тобто сховало б помилку налаштування за поганим результатом пошуку.
    """
    from pipeopt.solver.worker import WorkerTask, run_island_worker, set_simulator

    set_simulator(None)
    task = WorkerTask(diameters=[0.3, 0.4], v_opt=1.0, time_budget=1.0,
                      global_best_cost=float("inf"), seed=1, worker_id=0)

    with pytest.raises(RuntimeError, match="не ініціалізований"):
        run_island_worker(task)


def test_a_crashing_island_returns_an_empty_result_instead_of_propagating(capsys):
    """Аварія **всередині** пошуку — інша річ, ніж невдале налаштування.

    Острів може впасти на будь-якому кроці: EPANET не зійшовся, скінчилась
    пам'ять, дефект у стратегії. Тоді порожній результат — правильна
    відповідь: «цей острів нічого не дав», і оркестратор із цим працює.
    Але виняток мусить лишити слід у консолі, інакше прогін на N-1
    островах не відрізнити від нормального.
    """
    from pipeopt.solver.worker import WorkerTask, run_island_worker, set_simulator

    class BrokenSimulator:
        """Симулятор, який ламається на першому ж зверненні солвера."""

        n_variables = 3
        component_names = ["P1", "P2", "P3"]
        sources = ["R"]

        def __getattr__(self, name):
            raise RuntimeError(f"симулятор зламався на {name!r}")

    set_simulator(BrokenSimulator())
    task = WorkerTask(diameters=[0.3, 0.4], v_opt=1.0, time_budget=1.0,
                      global_best_cost=float("inf"), seed=1, worker_id=0)
    try:
        result = run_island_worker(task)
    finally:
        set_simulator(None)

    assert result.best_solution is None
    assert result.best_cost == float("inf")
    assert result.sim_count == 0
    assert "CRITICAL WORKER ERROR" in capsys.readouterr().out
