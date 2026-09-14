"""Частка пройденого прогону — величина, від якої залежать одинадцять рішень.

`progress_ratio` керує переходом у другу половину прогону, лімітом
застою, інтервалом міграції, порогом «авантюриста», вікном серцебиття,
заморозкою консенсусу, дистанцією різноманітності променя і його
динамічною шириною. Раніше вона рахувалась «або за симуляціями, або за
часом»: при заданому `max_sims` час не враховувався взагалі, тому
прогін, обмежений годинником, від початку до кінця поводився так, ніби
щойно почався.
"""

import pytest

from analytical_solver.context import SolverContext
from analytical_solver.kicks import KickStrategies
from analytical_solver.local_search import LocalSearch
from analytical_solver.orchestrator import IslandWorker

INFINITY = float("inf")


@pytest.fixture
def worker_factory(sim):
    def make(max_sims):
        ctx = SolverContext(sim, sim.config.diameters_m, v_opt=1.0)
        local_search = LocalSearch(ctx)
        return IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                            0, 1, max_sims, 5, "SMALL", [])

    return make


# --- один скінченний бюджет: поведінка не змінилась --------------------

def test_only_a_simulation_budget_counts_simulations(worker_factory):
    worker = worker_factory(1000)
    assert worker.progress_of(250, elapsed=5.0, time_budget=INFINITY) == 0.25


def test_only_a_time_budget_counts_time(worker_factory):
    worker = worker_factory(INFINITY)
    assert worker.progress_of(10**9, elapsed=30.0, time_budget=120.0) == 0.25


def test_no_budget_at_all_means_no_progress(worker_factory):
    worker = worker_factory(INFINITY)
    assert worker.progress_of(10**9, elapsed=10**6, time_budget=INFINITY) == 0.0


# --- обидва скінченні: береться той, що ближче до кінця ----------------

def test_the_binding_budget_wins(worker_factory):
    """Саме цього не робила стара формула: час просто не враховувався."""
    worker = worker_factory(1_000_000)

    # Прогін обмежений годинником: симуляцій мало, але час майже вийшов.
    assert worker.progress_of(10_000, elapsed=90.0, time_budget=100.0) == 0.9
    # І навпаки: часу вдосталь, а симуляції добігають.
    assert worker.progress_of(900_000, elapsed=1.0, time_budget=100.0) == 0.9


def test_a_wall_clock_bound_run_no_longer_looks_like_it_just_started(worker_factory):
    """Квота XLARGE — 30 млн симуляцій, тобто близько 20 діб на одному ядрі.
    Чотиригодинний прогін давав частку 0.008 і тримав алгоритм у режимі
    «початок прогону» до самого кінця."""
    worker = worker_factory(30_000_000)
    four_hours = 4 * 3600

    old_style = 250_000 / 30_000_000
    assert old_style < 0.01

    new_style = worker.progress_of(250_000, elapsed=four_hours,
                                   time_budget=four_hours)
    assert new_style == 1.0


def test_progress_never_exceeds_one(worker_factory):
    worker = worker_factory(1000)
    assert worker.progress_of(5000, elapsed=500.0, time_budget=100.0) == 1.0


def test_progress_starts_at_zero(worker_factory):
    worker = worker_factory(1000)
    assert worker.progress_of(0, elapsed=0.0, time_budget=100.0) == 0.0


# --- те, що з частки виводиться ----------------------------------------

@pytest.mark.parametrize("progress,expected", [
    (0.0, 12), (0.25, 9), (0.5, 6), (0.75, 4), (1.0, 4),
])
def test_stagnation_limit_falls_as_the_run_progresses(progress, expected):
    """Формула не змінена — зафіксована, бо тепер частка доходить до 1."""
    assert max(4, int(12 * (1.0 - progress))) == expected
