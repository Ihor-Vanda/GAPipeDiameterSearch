"""Єдиний модуль запуску воркера: контракт завдання й результату.

До етапу 4 запуск існував у трьох майже однакових реалізаціях, завдання
передавалося позиційним кортежем на 12 елементів, а симулятор діставався
з `__main__`. Тести фіксують новий контракт.
"""

import pytest

from analytical_solver.context import beam_width_for, classify_network
from analytical_solver.worker import (
    WorkerResult,
    WorkerTask,
    get_temp_root,
    run_island_worker,
    set_simulator,
)


@pytest.fixture
def task(sim):
    return WorkerTask(
        diameters=sim.config.diameters_m, v_opt=1.0, time_budget=1.0,
        global_best_cost=float("inf"), seed=1, worker_id=0,
    )


# --- класифікація мережі (раніше дублювалась у 4 місцях) -----------------

@pytest.mark.parametrize("num_pipes,expected", [
    (1, "SMALL"), (49, "SMALL"),
    (50, "MEDIUM"), (199, "MEDIUM"),
    (200, "LARGE"), (999, "LARGE"),
    (1000, "XLARGE"), (5000, "XLARGE"),
])
def test_classify_network_matches_legacy_boundaries(num_pipes, expected):
    assert classify_network(num_pipes) == expected


@pytest.mark.parametrize("network_class,width", [
    ("SMALL", 5), ("MEDIUM", 5), ("LARGE", 8), ("XLARGE", 8),
])
def test_beam_width_matches_legacy(network_class, width):
    assert beam_width_for(network_class) == width


# --- контракт завдання ---------------------------------------------------

def test_task_requires_only_the_essentials(sim):
    t = WorkerTask(diameters=sim.config.diameters_m, v_opt=1.0, time_budget=5.0,
                   global_best_cost=1.0, seed=7, worker_id=2)

    assert t.n_workers == 1
    assert t.max_sims == float("inf")
    assert t.shared_progress is None
    assert t.log_dir is None
    assert t.global_archive == []
    assert t.epoch == 0


def test_task_is_immutable(task):
    with pytest.raises(Exception):
        task.seed = 99


def test_each_task_gets_its_own_archive():
    a = WorkerTask(diameters=[], v_opt=1.0, time_budget=1.0, global_best_cost=1.0,
                   seed=1, worker_id=0)
    b = WorkerTask(diameters=[], v_opt=1.0, time_budget=1.0, global_best_cost=1.0,
                   seed=2, worker_id=1)
    a.global_archive.append("x")
    assert b.global_archive == [], "default_factory мусить давати окремий список"


# --- контракт результату -------------------------------------------------

def test_result_unpacks_positionally():
    cost, sol, sims, tabu = WorkerResult(1.5, [1, 2], 100, set())
    assert (cost, sol, sims) == (1.5, [1, 2], 100)
    assert tabu == set()


def test_result_has_no_placeholder_field():
    """Старий кортеж мав порожній третій елемент, який усі розпаковували в `_`."""
    assert WorkerResult._fields == ("best_cost", "best_solution", "sim_count", "basin_tabu")


# --- ініціалізація симулятора -------------------------------------------

def test_run_without_simulator_is_a_clear_error(task):
    set_simulator(None)
    with pytest.raises(RuntimeError, match="init_worker_process|set_simulator"):
        run_island_worker(task)


def test_set_simulator_enables_the_run(sim, task):
    set_simulator(sim)
    try:
        result = run_island_worker(task)
    finally:
        set_simulator(None)

    assert isinstance(result, WorkerResult)
    assert result.sim_count > 0


def test_temp_root_is_absolute():
    assert get_temp_root() == get_temp_root()
    import os
    assert os.path.isabs(get_temp_root())
