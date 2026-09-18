"""Оркестрація прогону: цикл моніторингу, запуск островів, фінальний полір.

`test_live_monitor.py` покрив **читання** стану островів — чисті функції
`_live_snapshot`, `_dedup_archive`, `_adopt_best`, `_worker_quota`. Цей
файл про решту `orchestrator.py`: про те, що читанням керує.

Чому воно лишалось непокритим і чому це погано:

* `_monitor_parallel` — сорок рядків, які виконуються **лише** в
  паралельному режимі. Еталон іде з `--sequential`, тобто не заходить
  сюди жодного разу; а щоб дістатись до них прогоном, потрібні реальні
  процеси й хвилини очікування. Тим часом саме цей цикл вирішує, що
  побачить користувач за годину пошуку і що потрапить у графік
  збіжності;
* паралельна гілка `_run_islands` ковтає виняток воркера й друкує
  `[Error] Worker N crashed`. Прогін із мертвим островом виглядає
  успішним: решта островів дадуть результат. Перевірити, що лічильники
  при цьому лишаються чесними, можна лише тут;
* `_final_polish` має гілку «жодного допустимого рішення не знайдено»,
  яка повертає запасний варіант. Вона виконується рідко — і саме тому
  ніколи не перевірялась.

Усе нижче працює без справжніх процесів: `_monitor_parallel` приймає
будь-що з методом `ready()`, `_run_islands` — будь-що з `apply_async`.
Це не полегшення тесту, а властивість коду: етап 15 виніс ці методи саме
тому, що вони не залежать від `multiprocessing`.
"""

import time

import pytest

from pipeopt.solver.orchestrator import AnalyticalSolver, RunState


@pytest.fixture
def solver(ctx):
    """Солвер із підставленим контекстом: жодного процесу не піднімається."""
    instance = AnalyticalSolver.__new__(AnalyticalSolver)
    instance.ctx = ctx
    instance.n_workers = 2
    instance.mp_pool = None
    instance.history = []
    instance.log_dir = None
    instance.max_sims = 100_000
    instance.time_limit_sec = 60.0
    return instance


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Цикл моніторингу спить секунду між обертами — тесту це ні до чого."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


class Pending:
    """Результат воркера, який стає готовим після `after` перевірок.

    `multiprocessing.pool.AsyncResult` у циклі використовується рівно
    через два методи — `ready()` і `get()`. Більшого тут не потрібно, і
    підміна лишається чесною.
    """

    def __init__(self, value=None, after=1, raises=None):
        self.value = value
        self.raises = raises
        self._left = after

    def ready(self):
        if self._left <= 0:
            return True
        self._left -= 1
        return False

    def get(self):
        if self.raises:
            raise self.raises
        return self.value


def progress(islands=None, **entries):
    """`shared_progress`: ключі островів — цілі, решта — рядки.

    Саме так його наповнює `solve_standalone` (`shared_progress[i] = 0`),
    тому підміна повторює цю форму, а не спрощує її.
    """
    state = dict(entries)
    state.update(islands or {})
    return state


# --- цикл моніторингу ---------------------------------------------------

def test_the_monitor_stops_as_soon_as_every_island_is_done(solver):
    """Умова виходу — готовність **усіх**, а не першого.

    Помилка тут не падає: цикл або обірветься на найшвидшому острові
    (решта дорахує у фоні, а їхні рішення пропадуть), або крутитиметься
    вічно.
    """
    results = [(0, Pending(after=1)), (1, Pending(after=3))]
    state = RunState()

    solver._monitor_parallel(results, progress(), state, time.time(), None)

    assert all(res.ready() for _, res in results)


def test_a_better_live_result_becomes_the_current_best(solver):
    """Рекорд оновлюється **під час** прогону, а не лише наприкінці.

    Від цього залежить, що покаже GUI за годину пошуку, і що піде у
    графік збіжності — файл, який потім переносять у дисертацію.
    """
    state = RunState(best_cost=9.0e6, best_sol=[3] * solver.ctx.num_pipes)
    shared = progress(global_best=(6.1e6, [1] * solver.ctx.num_pipes))

    solver._monitor_parallel([(0, Pending(after=1))], shared, state,
                             time.time(), None)

    assert state.best_cost == 6.1e6


def test_a_worse_live_result_does_not_touch_the_best(solver):
    state = RunState(best_cost=6.0e6, best_sol=[1] * solver.ctx.num_pipes)

    solver._monitor_parallel([(0, Pending(after=1))],
                             progress(global_best=(9.9e6, [3] * 6)),
                             state, time.time(), None)

    assert state.best_cost == 6.0e6


def test_the_convergence_history_gets_a_point(solver):
    """Історія — джерело графіка збіжності; порожня історія = порожній графік."""
    state = RunState()
    shared = progress({0: {"sims": 5000}},
                      global_best=(6.1e6, [1] * solver.ctx.num_pipes))

    solver._monitor_parallel([(0, Pending(after=1))], shared, state,
                             time.time(), None)

    assert solver.history
    evaluations, cost = solver.history[-1]
    assert cost == 6.1e6
    assert evaluations >= 5000, "у лічильник мусять увійти симуляції островів"


def test_nothing_is_recorded_while_no_island_has_a_solution(solver):
    """`inf` у графіку збіжності — це зіпсована вісь, а не точка."""
    state = RunState()

    solver._monitor_parallel([(0, Pending(after=1))], progress(), state,
                             time.time(), None)

    assert solver.history == []


def test_the_interface_is_told_about_progress(solver):
    """Кнопка «ЗУПИНИТИ» живе в цьому ж виклику: GUI кидає з нього виняток."""
    seen = []
    state = RunState()

    solver._monitor_parallel(
        [(0, Pending(after=1))],
        progress(global_best=(6.1e6, [1] * solver.ctx.num_pipes)),
        state, time.time(), lambda sims, cost, sol: seen.append((sims, cost, sol)))

    assert seen, "інтерфейс не отримав жодного оновлення"
    assert seen[-1][1] == 6.1e6


def test_the_solution_is_sent_to_the_interface_only_when_it_improves(solver):
    """Рішення — це список на сотні труб; слати його щосекунди марно.

    Тому воно передається лише разом із новим рекордом, а між рекордами
    йде `None`. Перевірка змістова: якби `None` не надсилався, GUI
    перемальовував би карту мережі щосекунди без причини.
    """
    seen = []
    state = RunState()
    shared = progress(global_best=(6.1e6, [1] * solver.ctx.num_pipes))

    solver._monitor_parallel([(0, Pending(after=3))], shared, state,
                             time.time(), lambda s, c, sol: seen.append(sol))

    assert seen[0] is not None, "перший рекорд мусить дійти з рішенням"
    assert all(sol is None for sol in seen[1:]), (
        "рішення надіслано повторно без нового рекорду")


def test_the_shared_archive_is_refreshed_for_the_islands(solver):
    """Архів — те, з чого острови беруть донорів для BASIN_ESCAPE.

    Якщо його не оновлювати, стратегія втечі з басейну працює з тим, що
    було на початку прогону.
    """
    shared = progress(global_best=(6.1e6, [1] * solver.ctx.num_pipes),
                      best_sol_0=(6.3e6, [2] * solver.ctx.num_pipes))

    solver._monitor_parallel([(0, Pending(after=1))], shared, RunState(),
                             time.time(), None)

    archive = shared["global_archive"]
    assert [cost for cost, _ in archive] == sorted(cost for cost, _ in archive)
    assert len(archive) == 2


# --- запуск островів ----------------------------------------------------

class FakePool:
    """Пул, який виконує завдання одразу й запам'ятовує їх."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.submitted = []

    def apply_async(self, _function, args):
        self.submitted.append(args[0])
        return self.outcomes.pop(0)


class WorkerResult:
    def __init__(self, best_cost, best_solution, sim_count=0):
        self.best_cost = best_cost
        self.best_solution = best_solution
        self.sim_count = sim_count


def test_every_island_result_is_collected(solver):
    solver.mp_pool = FakePool([
        Pending(WorkerResult(6.2e6, [1, 1, 1, 1, 1, 1]), after=1),
        Pending(WorkerResult(6.1e6, [2, 2, 2, 2, 2, 2]), after=1),
    ])
    tasks = solver._build_tasks(60.0, 1000, None, [])

    results = solver._run_islands(tasks, progress(), RunState(), time.time(), None)

    assert sorted(cost for cost, _ in results) == [6.1e6, 6.2e6]


def test_a_crashed_island_does_not_take_the_run_down(solver, capsys):
    """Пул ковтає виняток воркера — прогін із мертвим островом виглядає успішним.

    Тому важливі дві речі: решта островів дає результат, **і** про смерть
    повідомляється в консоль. Мовчазна втрата острова — це прогін на
    половину потужності, який нічим не відрізнити від нормального.
    """
    solver.mp_pool = FakePool([
        Pending(raises=RuntimeError("EPANET не зійшовся"), after=1),
        Pending(WorkerResult(6.1e6, [2, 2, 2, 2, 2, 2]), after=1),
    ])
    tasks = solver._build_tasks(60.0, 1000, None, [])

    results = solver._run_islands(tasks, progress(), RunState(), time.time(), None)

    assert len(results) == 1
    assert "crashed" in capsys.readouterr().out


def test_an_island_without_a_solution_is_skipped(solver):
    """Острів, який не знайшов нічого допустимого, не має що віддати."""
    solver.mp_pool = FakePool([Pending(WorkerResult(float("inf"), None), after=1)])
    solver.n_workers = 1
    tasks = solver._build_tasks(60.0, 1000, None, [])

    assert solver._run_islands(tasks, progress(), RunState(), time.time(), None) == []


def test_the_simulations_of_the_islands_are_added_up(solver):
    """Лічильник головного процесу їх не бачить: у кожного острова свій контекст.

    Без цього підсумкове «Total Sims» занижене в N разів, а вісь графіка
    збіжності бреше.
    """
    solver.mp_pool = FakePool([
        Pending(WorkerResult(6.1e6, [1] * 6), after=1),
        Pending(WorkerResult(6.2e6, [2] * 6), after=1),
    ])
    state = RunState()
    shared = progress({0: {"sims": 12_000}, 1: {"sims": 8_000}})

    solver._run_islands(solver._build_tasks(60.0, 1000, shared, []), shared,
                        state, time.time(), None)

    assert state.worker_sims == 20_000


def test_a_sequential_run_counts_simulations_from_the_result(solver):
    """Без `shared_progress` єдине джерело — те, що воркер повернув."""
    solver.mp_pool = None
    state = RunState()
    tasks = solver._build_tasks(60.0, 1000, None, [])

    import pipeopt.solver.orchestrator as orchestrator
    original = orchestrator.run_island_worker
    orchestrator.run_island_worker = lambda task: WorkerResult(
        6.1e6, [1] * 6, sim_count=7_000)
    try:
        results = solver._run_islands(tasks, None, state, time.time(), None)
    finally:
        orchestrator.run_island_worker = original

    assert state.worker_sims == 14_000, "два острови по 7 000"
    assert len(results) == 2


# --- завдання островам --------------------------------------------------

def test_each_island_gets_its_own_seed(solver):
    """Однакові зерна = N однакових пошуків замість острівної моделі."""
    tasks = solver._build_tasks(60.0, 5000, None, [])

    assert len({t.seed for t in tasks}) == len(tasks)
    assert [t.worker_id for t in tasks] == list(range(solver.n_workers))


def test_the_seeds_come_from_the_main_process_rng(solver):
    """Саме через це `--seed` доходить до воркерів (§12.4).

    `seed_modifier` береться з `random.randint` головного процесу, тому
    при засіяному RNG набір зерен островів відтворюваний.
    """
    import random

    random.seed(42)
    first = [t.seed for t in solver._build_tasks(60.0, 5000, None, [])]
    random.seed(42)
    second = [t.seed for t in solver._build_tasks(60.0, 5000, None, [])]

    assert first == second


def test_the_budget_and_the_archive_reach_every_island(solver):
    archive = [(6.1e6, [1] * 6)]

    tasks = solver._build_tasks(90.0, 5000, None, archive)

    assert all(t.max_sims == 5000 for t in tasks)
    assert all(t.time_budget == 90.0 for t in tasks)
    assert all(t.global_archive is archive for t in tasks)


# --- фінальний полір ----------------------------------------------------

def test_the_champion_is_polished_once_more(solver):
    """Останній глибокий спуск: кілька сотих відсотка, які ще лежать поруч."""
    from pipeopt.solver.local_search import LocalSearch

    solver.ls = LocalSearch(solver.ctx)
    state = RunState(best_sol=[3] * solver.ctx.num_pipes)
    state.best_cost = solver.ctx.get_cached_stats(state.best_sol)[0]
    before = state.best_cost

    solver._final_polish(state)

    assert state.best_cost <= before
    assert solver.ctx.get_cached_stats(state.best_sol)[1] >= solver.ctx.h_min


def test_polishing_records_the_final_point_of_the_curve(solver):
    from pipeopt.solver.local_search import LocalSearch

    solver.ls = LocalSearch(solver.ctx)
    state = RunState(best_sol=[3] * solver.ctx.num_pipes, worker_sims=50_000)
    state.best_cost = solver.ctx.get_cached_stats(state.best_sol)[0]

    solver._final_polish(state)

    assert solver.history[-1][0] >= 50_000


def test_a_run_that_found_nothing_returns_the_widest_pipes(solver, capsys):
    """Гілка, яка виконується рідко — і тому ніколи не перевірялась.

    Повернути `None` означало б впасти вже у звіті; повернути найбільші
    діаметри — дати завідомо допустиме (хоч і найдорожче) рішення й
    сказати про це в консоль.
    """
    state = RunState()

    solver._final_polish(state)

    assert state.best_sol == [solver.ctx.max_d_idx] * solver.ctx.num_pipes
    assert state.best_cost != float("inf")
    assert "No valid solution" in capsys.readouterr().out
