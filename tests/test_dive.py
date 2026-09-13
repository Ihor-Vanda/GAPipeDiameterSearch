"""SUBMARINE: політика допустимості та цикл занурення.

Головне, що тут перевіряється — режим `off` не додає жодної поступки.
Саме на цьому тримається твердження, що механізм не змінює поведінку,
поки його не увімкнули.
"""

import pytest

from analytical_solver.context import DEFICIT_PENALTY, SolverContext
from analytical_solver.dive import FEASIBILITY_MODES, FeasibilityPolicy


def test_off_never_relaxes_anything():
    p = FeasibilityPolicy(mode="off", depth=5.0)
    assert p.extra_tolerance() == 0.0
    p.diving = True                      # навіть якщо стан зіпсований
    assert p.extra_tolerance() == 0.0


def test_off_never_begins_a_dive():
    p = FeasibilityPolicy(mode="off")
    assert not p.maybe_begin(round_idx=100, stagnation=999, stag_limit=4,
                             best_cost=1.0, best_sol=[1, 2])
    assert not p.diving


def test_relax_is_always_on_and_never_surfaces():
    p = FeasibilityPolicy(mode="relax", depth=1.5)
    assert p.extra_tolerance() == 1.5
    assert not p.maybe_begin(0, 999, 4, 1.0, [1])
    assert not p.should_surface()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        FeasibilityPolicy(mode="periscope")


def test_all_modes_constructible():
    for mode in FEASIBILITY_MODES:
        assert FeasibilityPolicy(mode=mode).mode == mode


# --- цикл занурення ------------------------------------------------------

def make_dive(**kw):
    kw.setdefault("mode", "dive")
    kw.setdefault("depth", 2.0)
    kw.setdefault("max_rounds", 3)
    kw.setdefault("cooldown_rounds", 5)
    return FeasibilityPolicy(**kw)


def test_dive_requires_stagnation():
    p = make_dive(trigger_multiple=2.0)
    assert not p.maybe_begin(100, stagnation=7, stag_limit=4, best_cost=1.0, best_sol=[1])
    assert p.maybe_begin(100, stagnation=8, stag_limit=4, best_cost=1.0, best_sol=[1])


def test_dive_requires_a_rollback_point():
    p = make_dive()
    assert not p.maybe_begin(100, 999, 4, best_cost=1.0, best_sol=None)


def test_dive_stores_the_rollback_point():
    p = make_dive()
    sol = [1, 2, 3]
    p.maybe_begin(100, 999, 4, best_cost=7.5, best_sol=sol)

    assert p.surface_cost == 7.5
    assert p.surface_sol == sol
    assert p.surface_sol is not sol, "точку відкату треба копіювати"


def test_tolerance_is_on_only_while_diving():
    p = make_dive()
    assert p.extra_tolerance() == 0.0
    p.maybe_begin(100, 999, 4, 1.0, [1])
    assert p.extra_tolerance() == 2.0
    p.note_surfaced(110, feasible=True, new_cost=1.0, current_best=1.0)
    assert p.extra_tolerance() == 0.0


def test_surfaces_after_max_rounds():
    p = make_dive(max_rounds=3)
    p.maybe_begin(100, 999, 4, 1.0, [1])
    for _ in range(2):
        p.tick()
        assert not p.should_surface()
    p.tick()
    assert p.should_surface()


def test_cooldown_blocks_immediate_re_dive():
    p = make_dive(cooldown_rounds=5)
    p.maybe_begin(100, 999, 4, 1.0, [1])
    p.note_surfaced(100, feasible=True, new_cost=1.0, current_best=1.0)

    assert not p.maybe_begin(104, 999, 4, 1.0, [1]), "кулдаун не витримано"
    assert p.maybe_begin(105, 999, 4, 1.0, [1])


# --- облік заслуги -------------------------------------------------------

def test_credit_only_when_surfacing_beats_the_current_record():
    """Під час занурення звичайний пошук теж працює. Якщо рекорд покращив
    він, зараховувати це зануренню не можна."""
    p = make_dive()
    p.maybe_begin(100, 999, 4, best_cost=10.0, best_sol=[1])

    # виплили дешевше за точку занурення, але рекорд уже був кращим
    assert not p.note_surfaced(110, feasible=True, new_cost=9.0, current_best=8.0)
    assert p.stats.improved == 0
    assert p.stats.passive == 1
    assert p.stats.rolled_back == 1


def test_credit_when_surfacing_sets_a_new_record():
    p = make_dive()
    p.maybe_begin(100, 999, 4, best_cost=10.0, best_sol=[1])

    assert p.note_surfaced(110, feasible=True, new_cost=7.0, current_best=8.0)
    assert p.stats.improved == 1
    assert p.stats.total_gain == pytest.approx(1.0)
    assert p.stats.rolled_back == 0


def test_failed_surfacing_counts_as_rollback():
    p = make_dive()
    p.maybe_begin(100, 999, 4, 10.0, [1])

    assert not p.note_surfaced(110, feasible=False, new_cost=float("inf"), current_best=10.0)
    assert p.stats.surfaced == 0
    assert p.stats.rolled_back == 1


# --- поступка в ctx.score ------------------------------------------------

def test_score_forbids_deficit_beyond_tolerance(ctx):
    assert ctx.score(1000.0, ctx.h_min - 3.0, 10.0, tolerance=2.0) == float("inf")


def test_score_penalises_deficit_within_tolerance(ctx):
    deficit = 1.0
    expected = 1000.0 + deficit * 10.0 * DEFICIT_PENALTY
    assert ctx.score(1000.0, ctx.h_min - deficit, 10.0, tolerance=2.0) == pytest.approx(expected)


def test_deficit_is_more_expensive_than_any_surplus(ctx):
    """Занурення мусить бути дорогим, інакше пошук у ньому й залишиться."""
    diving = ctx.score(1000.0, ctx.h_min - 0.5, 10.0, tolerance=2.0)
    feasible = ctx.score(1000.0, ctx.h_min + 50.0, 10.0)
    assert diving > feasible


def test_zero_tolerance_reproduces_strict_behaviour(ctx):
    assert ctx.score(1000.0, ctx.h_min - 0.001, 10.0) == float("inf")
    assert ctx.score(1000.0, ctx.h_min - 0.001, 10.0, tolerance=0.0) == float("inf")


def test_ghost_shield_is_tolerance_aware(sim):
    """Без цього щит гасив кожне занурення: рішення з дефіцитом за
    побудовою виглядало «ілюзією»."""
    c = SolverContext(sim, sim.config.diameters_m)
    diving_sol = [0] * c.num_pipes           # недопустиме за тиском

    assert c.is_ghost_solution(diving_sol)
    assert not c.is_ghost_solution(diving_sol, tolerance=1000.0)
