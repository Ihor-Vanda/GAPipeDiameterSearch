"""Живий моніторинг островів: читання `shared_progress` без жодної симуляції.

Цей шматок був найбільшою дірою покриття в проєкті: сто рядків усередині
`solve_standalone`, які виконувались **лише** в паралельному прогоні на
кількох ядрах. Перевірити їх означало підняти пул процесів і чекати
хвилини, тому їх не перевіряв ніхто.

Після етапу 15 те саме читання живе в `_live_snapshot` і `_dedup_archive`
— чистих за наслідками функціях, яким потрібен звичайний словник. Тому
тести нижче виконуються за мілісекунди й не запускають ані процесів, ані
симуляцій.

Чому це варто тестувати. Моніторинг не приймає рішень про пошук, але
робить три речі, помилка в кожній з яких дорого коштує:

* обирає, **чиє** рішення показати як поточний рекорд — і це те саме
  число, яке бачить користувач у GUI й у логах;
* наповнює `global_archive`, з якого воркери беруть донорів для
  схрещування, тобто впливає на пошук опосередковано;
* веде `self.history`, з якої малюється графік збіжності у звіті.
"""

import pytest

from pipeopt.solver.orchestrator import AnalyticalSolver, RunState


@pytest.fixture
def solver(ctx):
    """Солвер із підставленим контекстом: пул не піднімається."""
    instance = AnalyticalSolver.__new__(AnalyticalSolver)
    instance.ctx = ctx
    instance.n_workers = 3
    instance.mp_pool = None
    instance.history = []
    return instance


# --- вибір поточного рекорда ------------------------------------------

def test_nothing_reported_yet_keeps_the_previous_best(solver):
    progress = {0: 0, 1: 0, 2: 0}

    best, sol, sims, parts, archive = solver._live_snapshot(progress, 9.0, [1, 2])

    assert (best, sol) == (9.0, [1, 2])
    assert sims == 0
    assert parts == ["W1:--", "W2:--", "W3:--"]
    assert archive == []


def test_a_worker_beating_the_current_best_takes_over(solver):
    progress = {0: {"sims": 5000, "best_cost": 7.5, "best_sol": [3, 3]},
                1: 0, 2: 0}

    best, sol, *_ = solver._live_snapshot(progress, 9.0, [1, 2])

    assert best == 7.5
    assert sol == [3, 3]


def test_a_worker_without_a_solution_still_moves_the_number(solver):
    """`best_cost` і `best_sol` приходять окремо — рішення може відставати.

    Тоді показане число оновлюється, а вектор лишається попереднім. Так
    було до розбору, і це зафіксовано навмисне: інакше в GUI поїхала б
    карта мережі, яка не відповідає підписаній під нею вартості.
    """
    progress = {0: {"sims": 100, "best_cost": 7.5}, 1: 0, 2: 0}

    best, sol, *_ = solver._live_snapshot(progress, 9.0, [1, 2])

    assert best == 7.5
    assert sol == [1, 2], "вектор не оновлюється, поки воркер його не надіслав"


def test_a_worse_worker_is_ignored(solver):
    progress = {0: {"sims": 10, "best_cost": 12.0, "best_sol": [0, 0]}, 1: 0, 2: 0}

    best, sol, *_ = solver._live_snapshot(progress, 9.0, [1, 2])

    assert (best, sol) == (9.0, [1, 2])


def test_the_global_best_is_read_before_the_workers(solver):
    """Порядок джерел значущий: воркер із тим самим числом перебиває глобал."""
    progress = {"global_best": (8.0, [5, 5]),
                0: {"sims": 1, "best_cost": 7.0, "best_sol": [6, 6]},
                1: 0, 2: 0}

    best, sol, *_ = solver._live_snapshot(progress, 9.0, None)

    assert (best, sol) == (7.0, [6, 6])


# --- лічильник симуляцій і рядок стану --------------------------------

def test_simulations_are_summed_across_workers(solver):
    progress = {0: {"sims": 1500}, 1: {"sims": 2500}, 2: {"sims": 0}}

    _, _, sims, _, _ = solver._live_snapshot(progress, 9.0, None)
    assert sims == 4000


def test_the_status_line_reports_thousands_and_a_dash_for_silence(solver):
    progress = {0: {"sims": 12_400}, 1: {"sims": 999}, 2: 0}

    _, _, _, parts, _ = solver._live_snapshot(progress, 9.0, None)
    assert parts == ["W1:12k", "W2:0k", "W3:--"]


def test_a_plain_number_is_accepted_instead_of_a_dict(solver):
    """До першого звіту воркера в `shared_progress` лежить просто нуль."""
    progress = {0: 7000, 1: 0, 2: 0}

    _, _, sims, parts, _ = solver._live_snapshot(progress, 9.0, None)
    assert sims == 7000
    assert parts[0] == "W1:7k"


# --- архів донорів ----------------------------------------------------

def test_the_archive_collects_the_global_best_and_every_worker(solver):
    progress = {"global_best": (8.0, [5, 5]),
                "best_sol_0": (7.0, [6, 6]), "best_sol_2": (9.0, [4, 4]),
                0: 0, 1: 0, 2: 0}

    _, _, _, _, archive = solver._live_snapshot(progress, float('inf'), None)

    assert sorted(c for c, _ in archive) == [7.0, 8.0, 9.0]


def test_the_archive_is_sorted_by_cost(solver):
    assert AnalyticalSolver._dedup_archive(
        [(9.0, [1]), (7.0, [2]), (8.0, [3])]) == [(7.0, [2]), (8.0, [3]), (9.0, [1])]


def test_identical_solutions_are_kept_once(solver):
    """Архів — джерело **різних** донорів; копії в ньому марні."""
    archive = AnalyticalSolver._dedup_archive(
        [(7.0, [1, 1]), (7.5, [1, 1]), (8.0, [2, 2])])

    assert [sol for _, sol in archive] == [[1, 1], [2, 2]]


def test_the_cheaper_copy_survives_deduplication(solver):
    archive = AnalyticalSolver._dedup_archive([(9.0, [1, 1]), (7.0, [1, 1])])

    assert archive == [(7.0, [1, 1])]


def test_the_archive_is_capped(solver):
    candidates = [(float(i), [i, i]) for i in range(20)]

    assert len(AnalyticalSolver._dedup_archive(candidates)) == 6
    assert len(AnalyticalSolver._dedup_archive(candidates, limit=2)) == 2


def test_an_empty_archive_stays_empty(solver):
    assert AnalyticalSolver._dedup_archive([]) == []


# --- збір підсумку з трьох джерел -------------------------------------

def test_a_worker_result_is_adopted(solver):
    state = RunState()

    solver._adopt_best(state, [(5.0, [1, 1])], None)

    assert state.best_cost == 5.0
    assert state.best_sol == [1, 1]


def test_the_sequential_run_has_no_other_source(solver):
    """`shared_progress is None` — саме цей випадок колись губив рішення.

    Регресія на §2.3.2: послідовна гілка повертала «всі максимальні»,
    бо результат воркера читався лише з `shared_progress`.
    """
    state = RunState()

    solver._adopt_best(state, [(5.0, [1, 1])], None)

    assert state.best_sol == [1, 1], "у послідовному режимі це єдине джерело"


def test_shared_progress_can_beat_the_returned_result(solver):
    state = RunState()
    progress = {"global_best": (4.0, [2, 2]), "best_sol_0": (6.0, [9, 9])}
    solver.n_workers = 1

    solver._adopt_best(state, [(5.0, [1, 1])], progress)

    assert state.best_cost == 4.0
    assert state.best_sol == [2, 2]


def test_a_worker_slot_can_beat_the_global_best(solver):
    """Останнє прочитане джерело перемагає — порядок збережено з оригіналу."""
    state = RunState()
    progress = {"global_best": (4.0, [2, 2]), "best_sol_0": (3.0, [7, 7])}
    solver.n_workers = 1

    solver._adopt_best(state, [], progress)

    assert state.best_cost == 3.0
    assert state.best_sol == [7, 7]


def test_the_adopted_solution_is_copied_not_aliased(solver):
    """Рішення приходить із чужого процесу — його не можна тримати за посиланням."""
    state = RunState()
    original = [1, 1]

    solver._adopt_best(state, [(5.0, original)], None)
    original[0] = 999

    assert state.best_sol == [1, 1]


def test_nothing_anywhere_leaves_the_state_empty(solver):
    state = RunState()

    solver._adopt_best(state, [], None)

    assert state.best_sol is None
    assert state.best_cost == float('inf')


# --- квота ------------------------------------------------------------

@pytest.mark.parametrize("max_sims,workers,expected", [
    (100_000, 4, 25_000),
    (100_000, 1, 100_000),
    (7, 2, 3),
])
def test_the_budget_is_split_between_islands(solver, max_sims, workers, expected):
    solver.max_sims, solver.n_workers = max_sims, workers

    assert solver._worker_quota() == expected


def test_an_unlimited_budget_stays_unlimited(solver):
    solver.max_sims, solver.n_workers = float('inf'), 4

    assert solver._worker_quota() == float('inf')
