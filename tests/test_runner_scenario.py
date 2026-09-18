"""Сценарій прогону: від `RunRequest` до файлів на диску.

`runner.py` — єдине місце, де сходяться всі частини: конфіг, симулятор,
солвер, звіти. До етапу 5 цей сценарій існував **двічі** (у `main()` і в
`gui.run_optimization`) і вже встиг розійтись у деталях. Після об'єднання
покриття лишалось 59 %: непокритими були саме `prepare`, `run`, `export`
— тобто весь наскрізний шлях.

Тести нижче проводять **справжні** прогони на Ханої з мікроскопічним
бюджетом (кількасот симуляцій замість мільйона). Це секунди, не хвилини,
і цього досить, щоб перевірити те, заради чого ці тести й пишуться:

* передпольотна перевірка **справді ловить неправильні одиниці** — і це
  виявилось не так, як записано в §30.4. Зонд брав найменші діаметри й
  перевіряв лише «чи не впав EPANET», а на найменших діаметрах тиск
  від'ємний і при правильних одиницях теж. Виправлено разом із цими
  тестами (етап 19);
* кнопка «ЗУПИНИТИ» справді зупиняє — GUI робить це винятком із
  `ui_callback`, і обробка цього винятку живе саме тут;
* серія з кількох прогонів обирає чемпіона за правилом «спершу
  допустимі, потім за вартістю», а не просто за вартістю;
* після `export` на диску лежить повний комплект файлів.
"""

import os

import pytest

from pipeopt.runner import OptimizationRunner, RunRequest, RunResult

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANOI = os.path.join(ROOT, "InputData", "Hanoi", "Hanoi.inp")
HANOI_COSTS = os.path.join(ROOT, "InputData", "Hanoi", "costs.csv")

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not os.path.exists(HANOI),
                                 reason="немає InputData/Hanoi")]


def tiny_request(**overrides):
    """Прогін на кількасот симуляцій: наскрізний шлях, а не якість пошуку."""
    params = dict(inp_file=HANOI, cost_file=HANOI_COSTS, h_min=30.0, units="in",
                  runs=1, cores=1, max_sims=300, run_mode="analytical",
                  sequential=True, time_limit=0)
    params.update(overrides)
    return RunRequest(**params)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Прогін пише робочі файли в поточний каталог — ізолюємо його."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- передпольотна перевірка -------------------------------------------

def test_preparing_a_valid_run_builds_the_cost_table(workspace):
    runner = OptimizationRunner(tiny_request())
    try:
        config = runner.prepare()

        assert config.diameters_m, "таблиця діаметрів мусить заповнитись"
        assert len(config.diameters_m) == len(config.diameters_raw)
        assert config.h_min == 30.0
    finally:
        runner.close()


def test_the_preflight_check_catches_the_wrong_units(workspace):
    """Ханої з `--units mm` — та сама помилка, що дає тиски −10⁸ (§18.5).

    Таблиця цін Ханої в дюймах; якщо тлумачити ті самі числа як
    міліметри, труби виходять у 25 разів вужчими.

    **Це не завжди ловилось.** До етапу 19 зонд брав найменші діаметри і
    перевіряв лише `simulation_ok`. На найменших діаметрах тиск
    від'ємний і при правильних одиницях теж (−1.8·10⁴ м на Ханої), тому
    обидва випадки виглядали однаково успішними, і прогін ішов десять
    хвилин перед тим, як дати нісенітницю.
    """
    runner = OptimizationRunner(tiny_request(units="mm"))
    try:
        with pytest.raises(RuntimeError, match="нерозв'язна"):
            runner.prepare()
    finally:
        runner.close()


def test_the_failure_message_names_all_three_likely_causes(workspace):
    """Повідомлення має вести до причини, а не лише констатувати провал."""
    runner = OptimizationRunner(tiny_request(units="mm"))
    try:
        with pytest.raises(RuntimeError) as failure:
            runner.prepare()
        message = str(failure.value)

        assert "--units mm" in message, "назвати поточний прапорець"
        assert "hmin=30" in message, "назвати поточний hmin"
        assert "costs.csv" in message, "назвати таблицю цін"
    finally:
        runner.close()


def test_an_unreachable_pressure_is_refused_before_the_run(workspace):
    """Та сама перевірка ловить і завідомо недосяжний hmin.

    Шукати годинами рішення задачі, яка не має розв'язку, безглуздо: якщо
    навіть найбільші діаметри не тримають тиск, жодне рішення не втримає.
    """
    runner = OptimizationRunner(tiny_request(h_min=500.0))
    try:
        with pytest.raises(RuntimeError, match="нерозв'язна"):
            runner.prepare()
    finally:
        runner.close()


def test_a_solvable_problem_passes_the_check(workspace):
    """Зворотний бік: правильна постановка не має відкидатись."""
    runner = OptimizationRunner(tiny_request())
    try:
        runner.prepare()
    finally:
        runner.close()


# --- наскрізний прогін --------------------------------------------------

@pytest.fixture
def completed(workspace):
    """Один справжній прогін із мікробюджетом."""
    runner = OptimizationRunner(tiny_request())
    runner.prepare()
    try:
        results = runner.run(str(workspace / "out"))
        yield runner, results, workspace / "out"
    finally:
        runner.close()


def test_a_run_produces_a_result(completed):
    _, results, _ = completed

    assert len(results) == 1
    assert isinstance(results[0], RunResult)


def test_the_result_carries_a_solution_for_every_pipe(completed):
    runner, results, _ = completed

    assert len(results[0].individual) == runner.simulator.n_variables
    assert all(isinstance(i, int) for i in results[0].individual)


def test_the_reported_cost_and_pressure_are_real_numbers(completed):
    _, results, _ = completed
    result = results[0]

    assert result.cost > 0 and result.cost != float("inf")
    assert result.pressure > 0
    assert result.feasible == (result.pressure >= 30.0)


def test_the_run_writes_its_log_where_it_was_told(completed):
    _, _, out_dir = completed

    assert (out_dir / "logs").exists() or any(out_dir.rglob("*.txt"))


# --- зупинка ------------------------------------------------------------

def test_a_run_that_is_stopped_returns_nothing(workspace):
    """Так поводиться кнопка «ЗУПИНИТИ»: перевірка перед кожним прогоном."""
    runner = OptimizationRunner(tiny_request(), should_stop=lambda: True)
    runner.prepare()
    try:
        assert runner.run(str(workspace / "out")) == []
    finally:
        runner.close()


def test_the_start_callback_fires_once_per_run(workspace):
    started = []
    runner = OptimizationRunner(tiny_request(runs=2),
                                on_run_start=started.append)
    runner.prepare()
    try:
        runner.run(str(workspace / "out"))
    finally:
        runner.close()

    assert started == [1, 2]


def test_several_runs_each_get_their_own_directory(workspace):
    runner = OptimizationRunner(tiny_request(runs=2))
    runner.prepare()
    try:
        runner.run(str(workspace / "out"))
    finally:
        runner.close()

    assert (workspace / "out" / "run_1").exists()
    assert (workspace / "out" / "run_2").exists()


# --- вибір чемпіона -----------------------------------------------------

def result(run_id, cost, feasible):
    return RunResult(run_id=run_id, cost=cost, pressure=30.0 if feasible else 10.0,
                     feasible=feasible, duration=1.0, individual=[0], history=[])


def test_the_cheapest_feasible_run_wins():
    champion = OptimizationRunner.champion(
        [result(1, 9.0, True), result(2, 7.0, True), result(3, 8.0, True)])

    assert champion.run_id == 2


def test_a_feasible_run_beats_a_cheaper_infeasible_one():
    """Недопустиме рішення не є рішенням, яким би дешевим воно не було."""
    champion = OptimizationRunner.champion(
        [result(1, 1.0, False), result(2, 9.0, True)])

    assert champion.run_id == 2


def test_the_cheapest_infeasible_run_wins_when_nothing_is_feasible():
    """Показати щось краще, ніж нічого — але воно позначене як FAIL."""
    champion = OptimizationRunner.champion(
        [result(1, 9.0, False), result(2, 7.0, False)])

    assert champion.run_id == 2
    assert champion.status == "FAIL"


def test_no_runs_means_no_champion():
    assert OptimizationRunner.champion([]) is None


# --- експорт ------------------------------------------------------------

def test_export_writes_the_whole_set_of_files(completed):
    runner, results, out_dir = completed

    best = runner.export(results, str(out_dir))

    assert best is not None
    assert (out_dir / "tables" / "solution_champion.csv").exists()
    assert (out_dir / "tables" / "optimized_network.inp").exists()
    assert (out_dir / "tables" / "engineering_report.txt").exists()
    assert (out_dir / "tables" / "runs_summary.csv").exists()
    assert (out_dir / "plots" / "network_map.png").exists()


def test_the_summary_has_a_row_per_run(completed):
    import pandas as pd

    runner, results, out_dir = completed
    runner.export(results, str(out_dir))

    summary = pd.read_csv(out_dir / "tables" / "runs_summary.csv")
    assert len(summary) == len(results)
    assert set(summary.columns) >= {"Run", "Cost", "Pressure", "Feasible", "Time"}


def test_exporting_nothing_is_not_an_error(completed):
    """Прогін, зупинений до першого результату, доходить сюди з порожнім списком."""
    runner, _, out_dir = completed

    assert runner.export([], str(out_dir)) is None


# --- прибирання ----------------------------------------------------------

def test_closing_the_runner_releases_the_simulator(workspace):
    """Поки симулятор живий, Windows не дає видалити його .rpt (§31.1)."""
    runner = OptimizationRunner(tiny_request())
    runner.prepare()
    assert runner.simulator is not None

    runner.close()

    assert runner.simulator is None


def test_closing_twice_is_safe(workspace):
    runner = OptimizationRunner(tiny_request())
    runner.prepare()

    runner.close()
    runner.close()


# --- кодування консолі: відтворення Windows на Linux --------------------

def test_a_console_that_cannot_show_emoji_does_not_kill_the_run(workspace):
    """Windows-дефект, відтворений тут без Windows.

    Прогін на машині користувача (Windows, Python 3.13, локаль cp1251)
    падав на **першому ж рядку** виводу солвера: у логах є `⚡`, а cp1251
    такого символу не має. `pytest` дав чотири падіння перевірки островів,
    а `reference_run.py --save` — жодного результату; корінь в обох
    випадках один.

    Вирішальна деталь — **коли** це спрацьовує. Інтерактивна консоль
    Windows часто вміє UTF-8, тому руками прогін ішов. Але
    `subprocess.run(..., capture_output=True)` робить stdout трубою, і
    для труби Python бере кодування локалі. Тобто падало саме те, що
    запускає прогін підпроцесом, — обидві автоматичні перевірки.

    `PYTHONIOENCODING=cp1251` відтворює ту саму умову на будь-якій
    системі, тому перевірка працює й там, де Windows немає. Без
    виправлення в `console.py` цей тест падає з кодом 1 і двома
    `UnicodeEncodeError` у виводі.
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"),
         "--inp", HANOI, "--costs", HANOI_COSTS, "--units", "in",
         "--cores", "1", "--sequential", "--seed", "42",
         "--max_sims", "3000", "--time_limit", "0"],
        cwd=str(workspace), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600,
        env=dict(os.environ, PYTHONIOENCODING="cp1251", MPLBACKEND="Agg"),
    )
    output = done.stdout + done.stderr

    assert "UnicodeEncodeError" not in output, (
        "консоль без емодзі зупинила прогін:\n" + output[-1500:])
    assert done.returncode == 0
    assert "FINAL RESULT" in output, "прогін не дійшов до результату"
