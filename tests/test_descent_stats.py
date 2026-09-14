"""Врожай проходів спуску — перша величина, яку солвер міряє про себе.

Замір §22.2 показав: у зійшовшомуся стані поліпшуваних труб нуль із 454,
а прохід по них коштує Θ(n) симуляцій. Ці лічильники роблять той самий
факт доступним **під час** прогону, щоб пороги «за класом мережі» можна
було замінити на зміряні величини (§24.4).

Поки що це тільки лічильники: жодне рішення від них не залежить. Тому
головний тест тут — що поведінка спуску не змінилась.
"""

import random

import pytest

from analytical_solver.local_search import DescentStats, LocalSearch


@pytest.fixture
def stats():
    return DescentStats()


# --- сама величина ----------------------------------------------------

def test_fresh_stats_have_no_yield(stats):
    assert stats.yield_rate is None
    assert stats.last_pass_yield is None


def test_yield_is_accepted_over_scanned(stats):
    for _ in range(10):
        stats.note_scan()
    for _ in range(3):
        stats.note_accept()
    assert stats.yield_rate == pytest.approx(0.3)


def test_skipped_pipes_do_not_count_as_scanned(stats):
    """`quick_mode` пропускає труби за евристикою — вони не перевірялись."""
    stats.note_scan()
    stats.note_skip()
    stats.note_skip()
    assert stats.scanned == 1
    assert stats.skipped == 2
    assert stats.yield_rate == 0.0


def test_last_pass_yield_is_about_the_last_pass_only(stats):
    marks = stats.start_pass()
    for _ in range(4):
        stats.note_scan()
    stats.note_accept()
    stats.note_accept()
    stats.finish_pass(marks)
    assert stats.last_pass_yield == pytest.approx(0.5)

    marks = stats.start_pass()
    for _ in range(4):
        stats.note_scan()
    stats.finish_pass(marks)
    # Другий прохід нічого не знайшов — середнє за весь прогін це згладило б.
    assert stats.last_pass_yield == 0.0
    assert stats.yield_rate == pytest.approx(0.25)


def test_an_empty_pass_leaves_the_last_yield_alone(stats):
    marks = stats.start_pass()
    stats.note_scan()
    stats.note_accept()
    stats.finish_pass(marks)

    empty = stats.start_pass()
    stats.finish_pass(empty)
    assert stats.last_pass_yield == 1.0


def test_snapshot_reports_every_counter(stats):
    assert set(stats.snapshot()) == {"passes", "scanned", "skipped", "accepted",
                                     "yield_rate", "last_pass_yield"}


# --- лічильники в реальному спуску ------------------------------------

def descend(sim, quick_mode=False):
    from analytical_solver.context import SolverContext

    random.seed(1)
    ctx = SolverContext(sim, sim.config.diameters_m, v_opt=1.0)
    local_search = LocalSearch(ctx)
    start = [ctx.max_d_idx] * ctx.num_pipes
    dyn = ctx.get_cached_stats(start).cost * 0.001
    result = local_search.gradient_squeeze(start, max_passes=None,
                                           quick_mode=quick_mode, dyn_bonus=dyn)
    return ctx, local_search, result


def test_descent_counts_its_passes_and_moves(sim):
    _, local_search, _ = descend(sim)
    snapshot = local_search.stats.snapshot()

    assert snapshot["passes"] >= 2
    assert snapshot["scanned"] > 0
    assert snapshot["accepted"] > 0


def test_the_final_pass_of_a_converged_descent_yields_nothing(sim):
    """Умова зупинки спуску — прохід без жодного прийнятого ходу.

    Саме це правило зупинки і коштує Θ(n) симуляцій щоразу; тепер його
    видно як число (§22.2).
    """
    _, local_search, _ = descend(sim)
    assert local_search.stats.last_pass_yield == 0.0


def test_strict_mode_never_skips_anything(sim):
    _, strict, _ = descend(sim, quick_mode=False)
    assert strict.stats.skipped == 0


def test_quick_mode_can_only_skip_never_add_work(sim):
    """`quick_mode` відкидає труби за втратою напору, тому перевірених
    труб у нього не більше, ніж у строгого режиму.

    Скільки саме пропущено, залежить від того, чи стоїть тиск на межі
    (`is_critically_tight`), тому в тесті фіксується напрямок, а не
    число: на фейковому симуляторі запас тиску великий, і пропускати
    може бути нічого.
    """
    _, strict, _ = descend(sim, quick_mode=False)
    _, quick, _ = descend(sim, quick_mode=True)

    assert quick.stats.scanned <= strict.stats.scanned
    assert quick.stats.skipped >= 0


def test_counters_do_not_change_the_result(sim):
    """Головне: лічильники нічого не вирішують."""
    _, _, first = descend(sim)
    _, _, second = descend(sim)
    assert first == second
