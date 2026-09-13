"""SolutionPool: точне tabu за відпечатком і tabu за підписом басейну."""

import pytest


@pytest.fixture
def pool(ctx):
    from analytical_solver.pool import SolutionPool

    return SolutionPool(ctx)


def test_exact_tabu_expires_after_tenure(pool, ctx):
    sol = [2] * ctx.num_pipes
    cost, _, _, _ = ctx.get_cached_stats(sol)

    pool.current_round = 0
    pool.add_to_tabu(sol, cost)

    assert pool.is_tabu(sol, cost, tenure=80)

    pool.current_round = 79
    assert pool.is_tabu(sol, cost, tenure=80), "у межах tenure має лишатись tabu"

    pool.current_round = 80
    assert not pool.is_tabu(sol, cost, tenure=80), "після tenure має звільнятись"


def test_unvisited_solution_is_not_tabu(pool, ctx):
    sol = [1] * ctx.num_pipes
    cost, _, _, _ = ctx.get_cached_stats(sol)
    assert not pool.is_tabu(sol, cost)


def test_hamming_distance(pool, ctx):
    a = [0] * ctx.num_pipes
    b = list(a)
    b[0] = 3
    b[2] = 1
    assert pool.hamming_distance(a, a) == 0
    assert pool.hamming_distance(a, b) == 2


def test_basin_signature_is_exact_for_small_networks(pool):
    """Для мереж <= 50 труб підпис басейну — це сам вектор рішення."""
    sol = [1, 2, 3, 0, 1, 2]
    assert pool.get_basin_signature(sol) == tuple(sol)


def test_add_basin_to_tabu_then_detected(pool):
    sol = [1, 2, 3, 0, 1, 2]
    assert not pool.is_basin_tabu(sol)
    pool.add_basin_to_tabu(sol)
    assert pool.is_basin_tabu(sol)


def test_clear_all_resets_every_memory(pool, ctx):
    sol = [2] * ctx.num_pipes
    cost, _, _, _ = ctx.get_cached_stats(sol)
    pool.active_pool.append((cost, cost, sol))
    pool.add_to_tabu(sol, cost)
    pool.add_basin_to_tabu(sol)
    pool.kick_tabu_set.add(("sig",))

    pool.clear_all()

    assert pool.active_pool == []
    assert pool.tabu_fingerprints == {}
    assert pool.kick_tabu_set == set()
    assert len(pool.basin_tabu) == 0


@pytest.fixture
def big_pool():
    """Пул для мережі > 50 труб — там підпис басейну вже не дорівнює вектору."""
    from conftest import FakeConfig, FakeSimulator

    from analytical_solver.context import SolverContext
    from analytical_solver.pool import SolutionPool

    big = FakeSimulator(n_pipes=60, config=FakeConfig())
    return SolutionPool(SolverContext(big, big.config.diameters_m))


def test_basin_signature_is_chunked_for_large_networks(big_pool):
    sig = big_pool.get_basin_signature([2] * 60)
    assert len(sig) < 60, "для великих мереж підпис має бути стиснутим"


def test_add_basin_to_tabu_works_for_large_networks(big_pool):
    sol = [2] * 60
    big_pool.add_basin_to_tabu(sol)
    assert big_pool.is_basin_tabu(sol)


def test_basin_escape_registers_signature_not_raw_vector(big_pool):
    """Етап 2: orchestrator тепер викликає add_basin_to_tabu(), а не
    basin_tabu.append(tuple(sol)).

    Раніше у deque клався повний вектор рішення, а is_basin_tabu() шукав там
    підпис басейну — для мереж > 50 труб це різні речі, тому basin tabu на
    Balerma не працював узагалі. Тест фіксує, що тепер записується підпис.
    """
    sol = [2] * 60
    big_pool.add_basin_to_tabu(sol)

    assert list(big_pool.basin_tabu) == [big_pool.get_basin_signature(sol)]
    assert big_pool.is_basin_tabu(sol)


def test_add_basin_to_tabu_deduplicates(big_pool):
    sol = [2] * 60
    big_pool.add_basin_to_tabu(sol)
    big_pool.add_basin_to_tabu(sol)

    assert len(big_pool.basin_tabu) == 1
