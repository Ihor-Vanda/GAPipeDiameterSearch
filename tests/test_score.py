"""SolverContext.score — єдина точка обчислення penalized score.

До етапу 3 ця формула існувала в коді у 13 місцях і у трьох різних
варіантах. Тести фіксують усі три політики премії за запас тиску.
"""

import pytest

from pipeopt.solver.context import (
    SURPLUS_DAMPING,
    SURPLUS_KNEE,
    SolverContext,
)


@pytest.fixture
def ctx_with(sim):
    def make(policy):
        return SolverContext(sim, sim.config.diameters_m, surplus_bonus_policy=policy)

    return make


def test_deficit_is_always_infinite(ctx):
    """Недопустиме за тиском рішення не може конкурувати за score."""
    assert ctx.score(cost=1000.0, p_min=ctx.h_min - 0.01, dyn_bonus=10.0) == float("inf")


def test_zero_surplus_scores_as_plain_cost(ctx):
    assert ctx.score(cost=1000.0, p_min=ctx.h_min, dyn_bonus=10.0) == 1000.0


def test_small_surplus_is_rewarded_identically_in_both_policies(ctx_with):
    """Нижче зламу damped і undamped мусять збігатися — саме тому уніфікація
    політики на Hanoi не змінила жодного з 8 прогонів."""
    small = SURPLUS_KNEE - 1.0
    damped = ctx_with("damped")
    undamped = ctx_with("undamped")
    p = damped.h_min + small

    assert damped.score(1000.0, p, 10.0) == undamped.score(1000.0, p, 10.0)
    assert damped.score(1000.0, p, 10.0) == 1000.0 - small * 10.0


def test_large_surplus_is_damped(ctx_with):
    big = SURPLUS_KNEE + 20.0
    damped = ctx_with("damped")
    undamped = ctx_with("undamped")
    p = damped.h_min + big

    assert undamped.score(1000.0, p, 10.0) == 1000.0 - big * 10.0
    assert damped.score(1000.0, p, 10.0) == 1000.0 - big * 10.0 * SURPLUS_DAMPING
    assert damped.score(1000.0, p, 10.0) > undamped.score(1000.0, p, 10.0), (
        "демпфування мусить робити великий запас менш привабливим"
    )


def test_none_policy_ignores_surplus(ctx_with):
    none = ctx_with("none")
    assert none.score(1000.0, none.h_min + 50.0, 10.0) == 1000.0
    assert none.score(1000.0, none.h_min - 0.01, 10.0) == float("inf")


def test_unknown_policy_is_rejected(sim):
    with pytest.raises(ValueError, match="surplus_bonus_policy"):
        SolverContext(sim, sim.config.diameters_m, surplus_bonus_policy="sharp")


def test_default_policy_is_damped(ctx):
    assert ctx.surplus_bonus_policy == "damped"
