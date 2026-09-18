"""«Щит від ілюзій»: коли підозріло великий виграш перевіряється симуляцією.

Перевірка стояла дослівно у двох місцях — `_apply_swap` і
`_beam_search_and_update`, — з однаковою неназваною константою 0.02 у
кожному. Тепер це `_is_ghost_gain`, і тут зафіксовано те, заради чого
перевірка написана саме так, а не інакше:

* `is_ghost_solution` **виконує симуляції**, тому вона не має права
  викликатись на кожен дрібний виграш — лише на підозріло великий;
* межа підозри названа (`GHOST_SUSPICION_FRACTION`), а не вписана двічі.

Другий пункт — не стиль: поки константа була записана в двох місцях,
зміна порогу в одному з них тихо розводила поведінку променя і свопу.
"""

import pytest

from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch
from pipeopt.solver.island import IslandWorker


class CountingGhostCheck:
    """Лічильник викликів `is_ghost_solution` із керованою відповіддю."""

    def __init__(self, ctx, answer):
        self.ctx = ctx
        self.answer = answer
        self.calls = 0

    def __call__(self, solution):
        self.calls += 1
        return self.answer


@pytest.fixture
def worker(ctx):
    local_search = LocalSearch(ctx)
    w = IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                     0, 1, 100_000, 5, [])
    w.run_best_sol = [2] * ctx.num_pipes
    w.run_best_cost = 1_000_000.0
    return w


def test_a_small_gain_is_never_paid_for_with_simulations(worker, ctx):
    probe = CountingGhostCheck(ctx, answer=True)
    ctx.is_ghost_solution = probe

    # виграш 1 % — рівно вдвічі менший за межу підозри
    assert worker._is_ghost_gain(990_000.0, [1] * ctx.num_pipes) is False
    assert probe.calls == 0, (
        "is_ghost_solution виконує симуляції; на дрібному виграші вона "
        "не має права викликатись")


def test_a_gain_exactly_on_the_threshold_is_still_trusted(worker, ctx):
    probe = CountingGhostCheck(ctx, answer=True)
    ctx.is_ghost_solution = probe
    on_threshold = worker.run_best_cost * (1 - IslandWorker.GHOST_SUSPICION_FRACTION)

    assert worker._is_ghost_gain(on_threshold, [1] * ctx.num_pipes) is False
    assert probe.calls == 0, "поріг нестрогий: рівно на межі перевірки немає"


def test_a_large_gain_is_checked_by_simulation(worker, ctx):
    probe = CountingGhostCheck(ctx, answer=True)
    ctx.is_ghost_solution = probe

    assert worker._is_ghost_gain(900_000.0, [1] * ctx.num_pipes) is True
    assert probe.calls == 1


def test_a_large_gain_that_survives_the_check_is_accepted(worker, ctx):
    ctx.is_ghost_solution = CountingGhostCheck(ctx, answer=False)

    assert worker._is_ghost_gain(900_000.0, [1] * ctx.num_pipes) is False


def test_a_more_expensive_solution_is_not_a_gain_at_all(worker, ctx):
    probe = CountingGhostCheck(ctx, answer=True)
    ctx.is_ghost_solution = probe

    assert worker._is_ghost_gain(1_200_000.0, [1] * ctx.num_pipes) is False
    assert probe.calls == 0


def test_the_threshold_is_named_once(island_source):
    """Регресія на дублювання: 0.02 більше не має бути вписана в код двічі.

    Сканується весь пакет `island`, а не один клас: після розділення на
    домішки `inspect.getsource(IslandWorker)` повертає лише тіло
    зібраного класу, і перевірка проходила б завжди.
    """
    source = island_source
    literal_uses = source.count("* 0.02)")
    assert literal_uses == 0, (
        "межа підозри мусить читатись із GHOST_SUSPICION_FRACTION, "
        f"а не вписуватись числом (знайдено {literal_uses} входжень)")
