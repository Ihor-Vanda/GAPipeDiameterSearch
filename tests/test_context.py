"""SolverContext: кеш, h_min і похідні структури."""


def test_h_min_is_exposed_directly(ctx, sim):
    """Прибирає ланцюг self.ctx.simulator.config.h_min через три об'єкти."""
    assert ctx.h_min == sim.config.h_min


def test_get_cached_stats_returns_named_result(ctx):
    st = ctx.get_cached_stats([2] * ctx.num_pipes)

    assert st.simulation_ok
    assert st.p_min <= st.p_max
    assert st.is_feasible(ctx.h_min) == (st.p_min >= ctx.h_min)


def test_get_cached_stats_counts_only_cache_misses(ctx):
    sol = [1] * ctx.num_pipes

    ctx.get_cached_stats(sol)
    after_first = ctx.sim_count
    ctx.get_cached_stats(sol)

    assert ctx.sim_count == after_first, "повторний запит мусить іти з кешу"


def test_graph_and_pipe_maps_are_built(ctx, sim):
    assert ctx.num_pipes == sim.n_variables
    assert ctx.base_G_flow.number_of_edges() == sim.n_variables
    assert len(ctx.edge_to_pipe) == 2 * sim.n_variables, "обидва напрямки ребра"
    assert ctx.max_d_idx == len(sim.config.diameters_m) - 1


def test_dominant_path_reaches_critical_node(ctx):
    sol = [2] * ctx.num_pipes
    st = ctx.get_cached_stats(sol)

    path_pipes, path_nodes = ctx.get_dominant_path(sol, st.crit_node)

    assert path_nodes[-1] == st.crit_node
    assert len(path_pipes) == len(path_nodes) - 1
