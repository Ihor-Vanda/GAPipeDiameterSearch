"""SimStats — іменований результат симуляції замість кортежу з пасткою."""

from pipeopt.hydraulics.sim_stats import SimStats


def test_unpacks_like_the_old_tuple():
    """Зворотна сумісність: старий код розпаковує результат позиційно."""
    cost, p_min, p_max, crit = SimStats(1000.0, 35.0, 80.0, "J7")

    assert (cost, p_min, p_max, crit) == (1000.0, 35.0, 80.0, "J7")


def test_third_field_is_max_pressure_not_a_flag():
    """Саме ця плутанина й була дефектом: третій елемент — p_max."""
    st = SimStats(1000.0, 35.0, 80.0, "J7")
    assert st.p_max == 80.0


def test_feasible_when_pressure_satisfied():
    st = SimStats(1000.0, 35.0, 80.0, "J7")
    assert st.simulation_ok
    assert st.is_feasible(30.0)
    assert st.surplus(30.0) == 5.0


def test_infeasible_when_pressure_below_limit():
    st = SimStats(1000.0, 25.0, 80.0, "J7")
    assert st.simulation_ok
    assert not st.is_feasible(30.0)
    assert st.surplus(30.0) == -5.0


def test_failed_simulation_is_never_feasible():
    """Раніше `if feas` на цьому результаті було ІСТИННИМ (p_max = -1.0)."""
    st = SimStats(1000.0, -1.0, -1.0, "ERR")

    assert not st.simulation_ok
    assert not st.is_feasible(30.0)
    assert bool(st.p_max), "p_max = -1.0 truthy — ось чому `if feas` нічого не перевіряв"
