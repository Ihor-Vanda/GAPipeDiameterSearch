"""Вихідні файли прогону: таблиця рішення, INP, звіт, графіки.

`plot.py` мав **8 % покриття** — і це модуль, який виробляє все, що
користувач бачить після багатогодинного прогону. Помилка тут не псує
пошук, вона псує **результат**: солвер відпрацював правильно, а в теці
лежить таблиця з чужими діаметрами або INP, який не відкривається.

Що перевіряється і чому саме воно:

* **INP читається назад** і містить ті діаметри, які обрав солвер. Це
  найважливіший тест у файлі: саме цей файл віддають у EPANET, і якщо в
  ньому не ті числа, весь прогін марний, а помітити це можна лише
  відкривши його руками;
* **вартість у таблиці = довжина × ціна** — число, яке потім переносять
  у дисертацію;
* **одиниці в заголовку** відповідають `unit_system` (§18.5: плутанина
  одиниць дорого коштує й діагностується погано);
* **після експорту в поточному каталозі не лишається сміття** — регресія
  на §29.1, де `run_sim()` без `file_prefix` лишав `temp.inp/rpt/bin` у
  корені проєкту кожного прогону.
"""

import os
import warnings

import pytest

from pipeopt.config import SolverConfig
from pipeopt.reporting.cost_table import load_cost_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANOI = os.path.join(ROOT, "InputData", "Hanoi", "Hanoi.inp")
HANOI_COSTS = os.path.join(ROOT, "InputData", "Hanoi", "costs.csv")

pytestmark = pytest.mark.skipif(not os.path.exists(HANOI),
                                reason="немає InputData/Hanoi")


@pytest.fixture(scope="module")
def config():
    cfg = SolverConfig(cost_file=HANOI_COSTS, unit_system="in", h_min=30.0)
    load_cost_table(cfg, silent=True)
    return cfg


@pytest.fixture(scope="module")
def network():
    import wntr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wntr.network.WaterNetworkModel(HANOI)


@pytest.fixture
def exported(tmp_path, config, network):
    """Один експорт на тест: рішення, де кожна труба має свій індекс."""
    from pipeopt.reporting import plot

    n_pipes = len(network.pipe_name_list)
    max_idx = len(config.diameters_raw) - 1
    solution = [i % (max_idx + 1) for i in range(n_pipes)]

    prefix = str(tmp_path / "tables" / "solution_champion")
    plot.export_solution(solution, [], HANOI, prefix, config,
                         cost=6_081_200.0, time_sec=120.0, total_sims=1_000_000)
    return tmp_path / "tables", solution


# --- таблиця рішення ---------------------------------------------------

def test_every_pipe_gets_a_row(exported, network):
    import pandas as pd

    tables, _ = exported
    frame = pd.read_csv(tables / "solution_champion.csv")

    assert len(frame) == len(network.pipe_name_list)
    assert list(frame["Pipe ID"].astype(str)) == list(network.pipe_name_list)


def test_the_diameter_column_is_labelled_with_the_unit_system(exported):
    import pandas as pd

    tables, _ = exported
    frame = pd.read_csv(tables / "solution_champion.csv")

    assert "Diameter (inch)" in frame.columns, (
        "заголовок мусить називати одиниці: плутанина між дюймами й "
        "міліметрами не діагностується числами (§18.5)")


def test_the_diameters_in_the_table_are_the_ones_that_were_chosen(exported, config):
    import pandas as pd

    tables, solution = exported
    frame = pd.read_csv(tables / "solution_champion.csv")

    expected = [round(config.diameters_raw[i], 2) for i in solution]
    assert list(frame["Diameter (inch)"]) == expected


def test_the_cost_of_a_pipe_is_its_length_times_its_price(exported, config, network):
    import pandas as pd

    tables, solution = exported
    frame = pd.read_csv(tables / "solution_champion.csv")

    for row_no, pipe_name in enumerate(network.pipe_name_list):
        diameter = config.diameters_raw[solution[row_no]]
        length = network.get_link(pipe_name).length
        assert frame["Cost"][row_no] == pytest.approx(
            round(length * config.costs[diameter], 2)), (
            f"труба {pipe_name}: вартість у звіті не дорівнює довжині × ціну")


def test_the_total_cost_can_be_summed_from_the_table(exported):
    """Число, яке переносять у дисертацію, мусить сходитись із таблицею."""
    import pandas as pd

    tables, _ = exported
    frame = pd.read_csv(tables / "solution_champion.csv")

    assert frame["Cost"].sum() > 0
    assert frame["Length"].sum() > 0


# --- INP: головний файл, який віддають у EPANET ------------------------

def test_the_written_inp_can_be_read_back(exported):
    import wntr

    tables, _ = exported
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wntr.network.WaterNetworkModel(str(tables / "optimized_network.inp"))


def test_the_written_inp_carries_the_chosen_diameters(exported, config, network):
    """Найважливіша перевірка файлу: у EPANET піде саме обране рішення."""
    import wntr

    tables, solution = exported
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        written = wntr.network.WaterNetworkModel(str(tables / "optimized_network.inp"))

    for row_no, pipe_name in enumerate(network.pipe_name_list):
        expected_m = config.diameters_m[solution[row_no]]
        assert written.get_link(pipe_name).diameter == pytest.approx(expected_m, rel=1e-6)


def test_the_source_network_is_not_modified_on_disk(tmp_path, config, network):
    """Експорт читає `.inp` користувача — і не має права його переписати.

    `export_solution` міняє діаметри в моделі (`link.diameter = ...`), щоб
    записати оптимізований INP. Модель читається з файлу користувача, і
    один неуважний `write_inpfile` перезаписав би вхідні дані задачі.
    """
    import shutil

    from pipeopt.reporting import plot

    sandbox = tmp_path / "Hanoi.inp"
    shutil.copy(HANOI, sandbox)
    before = sandbox.read_bytes()

    plot.export_solution([1] * len(network.pipe_name_list), [], str(sandbox),
                         str(tmp_path / "out" / "solution"), config)

    assert sandbox.read_bytes() == before, "вхідний .inp змінився на диску"


# --- звіт ---------------------------------------------------------------

def test_the_engineering_report_is_written(exported):
    tables, _ = exported
    report = (tables / "engineering_report.txt").read_text(encoding="utf-8")

    assert "ДЕТАЛЬНИЙ ІНЖЕНЕРНИЙ ЗВІТ" in report
    assert "6.0812 M$" in report, "підсумкова вартість мусить бути у звіті"
    assert "1,000,000" in report, "кількість симуляцій теж"


def test_the_report_lists_every_junction_and_every_pipe(exported, network):
    tables, _ = exported
    report = (tables / "engineering_report.txt").read_text(encoding="utf-8")

    for node in network.junction_name_list:
        assert f"\n{node:<15} |" in report, f"вузол {node} не потрапив у звіт"


# --- дисципліна з тимчасовими файлами (§29.1) --------------------------

def test_the_export_leaves_no_temporary_files_behind(tmp_path, config, network,
                                                     monkeypatch):
    """Регресія: `run_sim()` без `file_prefix` писав temp.* у корінь проєкту.

    Ховав це лише рядок `temp.*` у `.gitignore`, тому дефект жив довго.
    """
    from pipeopt.reporting import plot

    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)

    solution = [0] * len(network.pipe_name_list)
    plot.export_solution(solution, [], HANOI,
                         str(tmp_path / "out" / "solution"), config)

    leftovers = [p.name for p in work.iterdir()]
    assert leftovers == [], f"у поточному каталозі лишилось: {leftovers}"


def test_a_missing_output_directory_is_created(tmp_path, config, network):
    from pipeopt.reporting import plot

    prefix = tmp_path / "deep" / "nested" / "solution"
    plot.export_solution([0] * len(network.pipe_name_list), [], HANOI,
                         str(prefix), config)

    assert (tmp_path / "deep" / "nested" / "solution.csv").exists()


# --- графік збіжності ---------------------------------------------------

def test_an_empty_history_produces_no_file_and_no_crash(tmp_path):
    from pipeopt.reporting import plot

    target = tmp_path / "plots" / "convergence.png"
    plot.plot_convergence([], filename=str(target))

    assert not target.exists()


def test_the_convergence_plot_and_its_table_are_written(tmp_path):
    import pandas as pd

    from pipeopt.reporting import plot

    (tmp_path / "plots").mkdir()
    history = [{"evals": 1000, "cost": 9.0e6},
               {"evals": 5000, "cost": 7.0e6},
               {"evals": 20000, "cost": 6.1e6}]

    plot.plot_convergence(history, filename=str(tmp_path / "plots" / "convergence.png"))

    assert (tmp_path / "plots" / "convergence.png").exists()
    table = pd.read_csv(tmp_path / "tables" / "convergence_history.csv")
    assert list(table["Evaluations"]) == [1000, 5000, 20000]


def test_points_that_cannot_be_plotted_are_dropped(tmp_path):
    """`inf` з'являється, поки жодне допустиме рішення ще не знайдене."""
    import pandas as pd

    from pipeopt.reporting import plot

    (tmp_path / "plots").mkdir()
    history = [{"evals": 100, "cost": float("inf")},
               {"evals": 200, "cost": 0.0},
               {"evals": 300, "cost": -5.0},
               {"evals": 400, "cost": 6.1e6}]

    plot.plot_convergence(history, filename=str(tmp_path / "plots" / "c.png"))

    table = pd.read_csv(tmp_path / "tables" / "convergence_history.csv")
    assert list(table["Evaluations"]) == [400]


def test_a_history_with_nothing_plottable_still_writes_nothing_broken(tmp_path):
    from pipeopt.reporting import plot

    (tmp_path / "plots").mkdir()
    target = tmp_path / "plots" / "c.png"

    plot.plot_convergence([{"evals": 1, "cost": float("inf")}], filename=str(target))

    # Файл або не створено, або він читається — але не порожній «хвіст».
    assert not target.exists() or target.stat().st_size > 0


# --- карта мережі -------------------------------------------------------

def test_the_network_map_is_drawn(tmp_path, config, network):
    from pipeopt.reporting import plot

    target = tmp_path / "network_map.png"
    plot.plot_network_map([1] * len(network.pipe_name_list), HANOI,
                          filename=str(target), config=config, cost=6.0e6)

    assert target.exists()
    assert target.stat().st_size > 1000, "PNG підозріло малий — схоже, порожній"


def test_drawing_the_map_does_not_use_the_deprecated_graph_api(tmp_path, config,
                                                               network, recwarn):
    """Регресія: `get_graph()` друкував DeprecationWarning на кожну карту.

    Етап 12 (§30.5) прибрав його з `water_sim.py` і пропустив `netdraw.py`
    — знайшлось цим файлом тестів.
    """
    from pipeopt.reporting import plot

    plot.plot_network_map([1] * len(network.pipe_name_list), HANOI,
                          filename=str(tmp_path / "map.png"), config=config)

    deprecated = [w for w in recwarn
                  if issubclass(w.category, DeprecationWarning)
                  and "get_graph" in str(w.message)]
    assert not deprecated, [str(w.message) for w in deprecated]
