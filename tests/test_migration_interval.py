"""Інтервал міграції: коли воркер знову дивиться на сусідів.

Розрахунок жив у двох місцях `run()` і був записаний двома різними
виразами — `max_sims // 100` перед циклом і `(max_sims // 20) // 5`
всередині. Це те саме число, але доводити це доводилось олівцем. Крім
того, блок перед циклом не мав ефекту взагалі: `progress_ratio` на той
момент завжди 0.0, тож гілка «кінець прогону» була недосяжна, а решта
перераховувалась у циклі до першого використання.

Тут зафіксовані властивості, які мусять лишатись правдою після будь-якого
наступного переписування: інтервал не падає нижче межі, у другій половині
прогону скорочується, і при необмеженому бюджеті симуляцій не залежить
від прогресу.
"""

import pytest

from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch
from pipeopt.solver.island import IslandWorker


@pytest.fixture
def worker_factory(ctx):
    def make(max_sims):
        local_search = LocalSearch(ctx)
        return IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                            0, 4, max_sims, 5, [])
    return make


def test_an_unlimited_budget_gives_the_fixed_default(worker_factory):
    w = worker_factory(float('inf'))
    for ratio in (0.0, 0.5, 0.95):
        w.progress_ratio = ratio
        assert w._migration_interval() == IslandWorker.DEFAULT_MIGRATION_INTERVAL


def test_the_interval_never_drops_below_the_floor(worker_factory):
    """Крихітний бюджет не має перетворювати міграцію на щороундову."""
    w = worker_factory(1000)
    for ratio in (0.0, 0.5, 0.99):
        w.progress_ratio = ratio
        assert w._migration_interval() >= IslandWorker.MIN_MIGRATION_INTERVAL


def test_the_interval_shrinks_as_the_run_progresses(worker_factory):
    w = worker_factory(10_000_000)
    intervals = []
    for ratio in (0.0, 0.25, 0.5, 0.75):
        w.progress_ratio = ratio
        intervals.append(w._migration_interval())
    assert intervals == sorted(intervals, reverse=True), (
        "що ближче до кінця прогону, то частіше воркери синхронізуються")


def test_the_last_fifth_of_the_run_synchronises_five_times_more_often(worker_factory):
    w = worker_factory(10_000_000)
    w.progress_ratio = IslandWorker.LATE_MIGRATION_RATIO + 0.01
    late = w._migration_interval()
    assert late == max(IslandWorker.MIN_MIGRATION_INTERVAL,
                       (10_000_000 // 20) // 5)


def test_the_late_branch_is_reachable_at_all(worker_factory):
    """Регресія: до правки цю гілку рахували на `progress_ratio == 0.0`.

    Тобто вона була недосяжна за побудовою. Тест ловить повернення до
    розрахунку інтервалу до того, як прогрес узагалі відомий.
    """
    w = worker_factory(10_000_000)
    w.progress_ratio = 0.0
    early = w._migration_interval()
    w.progress_ratio = 0.9
    late = w._migration_interval()
    assert late < early


def test_a_fresh_worker_starts_with_the_default_interval(worker_factory):
    """Об'єкт мусить бути валідним до виклику `run()`."""
    assert worker_factory(float('inf'))._migration_interval_sims == \
        IslandWorker.DEFAULT_MIGRATION_INTERVAL
