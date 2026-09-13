"""LocalSearch на фейковому симуляторі — без EPANET, за мілісекунди."""

import pytest


@pytest.fixture
def ls(ctx):
    from analytical_solver.local_search import LocalSearch

    return LocalSearch(ctx)


def test_gradient_squeeze_reduces_cost_and_keeps_feasibility(ls, ctx):
    start = [ctx.max_d_idx] * ctx.num_pipes
    cost_before, _, _, _ = ctx.get_cached_stats(start)

    result = ls.gradient_squeeze(start, quick_mode=False)
    cost_after, p_after, _, _ = ctx.get_cached_stats(result)

    assert cost_after < cost_before, "спуск мусить дешевшати"
    assert p_after >= ctx.simulator.config.h_min, "і не порушувати тиск"


def test_gradient_squeeze_respects_locked_pipes(ls, ctx):
    start = [ctx.max_d_idx] * ctx.num_pipes
    locked = {0, 1}

    result = ls.gradient_squeeze(start, locked_pipes=locked, quick_mode=False)

    for i in locked:
        assert result[i] == start[i], f"труба {i} заблокована, її не можна змінювати"


def test_gradient_squeeze_does_not_mutate_input(ls, ctx):
    start = [ctx.max_d_idx] * ctx.num_pipes
    snapshot = list(start)

    ls.gradient_squeeze(start, quick_mode=False)

    assert start == snapshot, "вхідний список має лишатись незмінним"


def test_classical_descent_ignores_limits_and_locks(ls, ctx):
    """Режим ablation: quick_mode/max_passes/locked_pipes мають бути перекриті."""
    start = [ctx.max_d_idx] * ctx.num_pipes

    limited = ls.gradient_squeeze(start, locked_pipes={0, 1, 2}, max_passes=1, quick_mode=True)
    classical = ls.gradient_squeeze(start, locked_pipes={0, 1, 2}, max_passes=1, quick_mode=True,
                                    classical_descent=True)

    c_limited, _, _, _ = ctx.get_cached_stats(limited)
    c_classical, p_classical, _, _ = ctx.get_cached_stats(classical)

    assert c_classical <= c_limited, "класичний спуск не має бути гіршим"
    assert p_classical >= ctx.simulator.config.h_min
    assert classical != limited, "блокування мали бути проігноровані"


def test_heal_network_repairs_infeasible_solution(ls, ctx):
    broken = [0] * ctx.num_pipes
    _, p_before, _, _ = ctx.get_cached_stats(broken)
    assert p_before < ctx.simulator.config.h_min, "стартове рішення має бути недопустимим"

    healed, ok, boosts = ls.heal_network(list(broken), set())

    assert ok, "мережа мусить відновитись"
    assert boosts > 0, "відновлення вимагає підняття діаметрів"
    _, p_after, _, _ = ctx.get_cached_stats(healed)
    assert p_after >= ctx.simulator.config.h_min


def test_heal_network_never_touches_locked_pipes(ls, ctx):
    broken = [0] * ctx.num_pipes
    locked = {0}

    healed, _, _ = ls.heal_network(list(broken), locked)

    assert healed[0] == 0, "заблокована труба не може бути піднята"


def test_evaluate_candidate_rejects_when_nothing_to_change(ls, ctx):
    at_max = [ctx.max_d_idx] * ctx.num_pipes
    score, cost, sol = ls.evaluate_candidate(at_max, [0], "upgrade", dyn_bonus=1.0)

    assert sol is None, "апгрейд труби на максимумі неможливий"
    assert score == float("inf")


def test_get_high_impact_pipes_prefers_expensive_loaded_pipes(ls, ctx):
    sol = [ctx.max_d_idx] * ctx.num_pipes
    top = ls.get_high_impact_pipes(sol, top_k=3)

    assert len(top) == 3
    assert len(set(top)) == 3, "індекси не повинні дублюватись"
    assert all(0 <= i < ctx.num_pipes for i in top)
