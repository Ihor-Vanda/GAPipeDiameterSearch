"""Наскрізна регресія солвера на фейковому симуляторі.

Це та сама «страхова сітка», що описана як етап 0 плану рефакторингу, але
без EPANET: `IslandWorker` проганяється на аналітичній мережі й перевіряється
детермінізм і якість результату. Виконується за секунди, тому може стояти в
CI і запускатись після кожного етапу.

Примітка про `__main__.worker_sim_instance`: послідовна гілка
`solve_standalone` викликає `AnalyticalSolver.worker_task`, який дістає
симулятор із простору імен `__main__`. Тест мусить це імітувати — і саме це
незручність, яку прибирає етап 4 (винесення запуску воркера в
`analytical_solver/worker.py` з явною передачею симулятора).
"""

import __main__
import random

import numpy as np
import pytest

from analytical_solver import AnalyticalSolver

pytestmark = pytest.mark.slow


@pytest.fixture
def solver_factory(sim):
    def make(seed, max_sims=2500, time_limit_sec=4):
        random.seed(seed)
        np.random.seed(seed)
        __main__.worker_sim_instance = sim
        return AnalyticalSolver(
            sim, sim.config.diameters_m, v_opt=1.0,
            max_sims=max_sims, time_limit_sec=time_limit_sec,
            pool=None, log_dir=None, n_workers=1,
        )

    yield make
    if hasattr(__main__, "worker_sim_instance"):
        del __main__.worker_sim_instance


def _cost_of(sim, real_diameters):
    idx = [min(range(sim.n_options), key=lambda j: abs(sim.diams_m[j] - d))
           for d in real_diameters]
    return sim.get_stats(idx)


def test_solve_standalone_returns_feasible_solution(solver_factory, sim):
    diameters = solver_factory(seed=1).solve_standalone()

    assert len(diameters) == sim.n_variables
    _, p_min, _, _ = _cost_of(sim, diameters)
    assert p_min >= sim.config.h_min, "результат мусить задовольняти обмеження тиску"


def test_solve_standalone_beats_all_max_baseline(solver_factory, sim):
    diameters = solver_factory(seed=1).solve_standalone()

    cost, _, _, _ = _cost_of(sim, diameters)
    baseline, _, _, _ = sim.get_stats([sim.n_options - 1] * sim.n_variables)
    assert cost < baseline, "оптимізація мусить бути дешевшою за «всі максимальні»"


def test_same_seed_gives_identical_result(solver_factory):
    """Головна властивість для порівняння етапів рефакторингу."""
    first = solver_factory(seed=12345).solve_standalone()
    second = solver_factory(seed=12345).solve_standalone()

    assert first == second, (
        "однаковий seed мусить давати однакове рішення; якщо впало — "
        "десь у солвері лишилось незасіяне джерело випадковості"
    )


def test_solve_fast_is_reproducible(solver_factory, sim):
    first = solver_factory(seed=77).solve_fast()
    second = solver_factory(seed=77).solve_fast()

    assert first == second
    _, p_min, _, _ = _cost_of(sim, first)
    assert p_min >= sim.config.h_min
