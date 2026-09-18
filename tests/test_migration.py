"""Міграція між островами: коли воркер дивиться на сусідів і що з цим робить.

До етапу 16 це були 63 рядки на шостому рівні відступу всередині `run()`
— найглибше вкладений і найменш перевірений шматок циклу. §4.4 плану
вимагав винести його ще на етапі 4; тести з'явились разом із виносом.

Перевіряти тут є що: міграція **єдина**, крім кіків, може підмінити
власний рекорд воркера чужим рішенням. Помилка в її умовах не падає й не
логується — вона просто стягує всі острови в одну точку, і острівна
модель тихо перетворюється на один пошук у N копіях.

Три політики обирає температура застою:

* `T < 0.3` — холодний острів: гібрид іде **в пул**, рекорд лишається своїм;
* `T < 0.7` — теплий: гібрид може стати новим рекордом;
* інакше — щит: сусідів ігноруємо, бо острів у глибокому пошуку.
"""

import random

import pytest

from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch
from pipeopt.solver.island import IslandWorker


@pytest.fixture
def worker(ctx):
    random.seed(5)
    local_search = LocalSearch(ctx)
    w = IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                     worker_id=0, n_workers=3, max_sims=100_000, beam_width=5, global_archive=[])
    w.run_best_sol = [2] * ctx.num_pipes
    w.run_best_cost = ctx.get_cached_stats(w.run_best_sol).cost
    w.base_dyn_bonus = w.run_best_cost * 0.001
    w.pool.active_pool = [(0.0, w.run_best_cost, list(w.run_best_sol))]
    # інтервал уже вичерпано: міграція має право спрацювати
    w._last_migration_sims = -10**9
    w._migration_interval_sims = 1
    return w


def global_best(worker, fraction, sol=None):
    """Глобальний рекорд, дешевший за власний у `fraction` разів."""
    return (worker.run_best_cost * fraction,
            sol if sol is not None else [1] * len(worker.run_best_sol))


# --- температура: одне джерело ----------------------------------------

def test_temperature_is_zero_without_stagnation(worker):
    worker.stagnation_counter = 0
    assert worker.temperature == 0.0


def test_temperature_saturates_at_one(worker):
    worker.stag_limit, worker.stagnation_counter = 4, 10_000
    assert worker.temperature == 1.0


def test_temperature_grows_with_stagnation(worker):
    worker.stag_limit = 4
    seen = []
    for counter in (0, 3, 6, 12):
        worker.stagnation_counter = counter
        seen.append(worker.temperature)
    assert seen == sorted(seen)


def test_the_formula_lives_in_one_place(island_source):
    """Регресія: раніше вона була дослівно у двох місцях під різними іменами."""
    source = island_source
    assert source.count("self.stag_limit * 3.0") == 1, (
        "температура мусить рахуватись лише у властивості `temperature`")


# --- коли міграції не відбувається ------------------------------------

def test_a_sequential_run_never_migrates(worker):
    """`shared_progress is None` — сусідів просто немає."""
    before = list(worker.pool.active_pool)

    worker._maybe_migrate(global_best(worker, 0.5), None)

    assert worker.pool.active_pool == before


def test_the_interval_holds_the_worker_back(worker):
    worker._last_migration_sims = worker.ctx.sim_count
    worker._migration_interval_sims = 10**9
    before = list(worker.pool.active_pool)

    worker._maybe_migrate(global_best(worker, 0.5), {})

    assert worker.pool.active_pool == before, (
        "поки інтервал не вичерпано, воркер на сусідів не дивиться")


def test_a_worker_that_is_not_lagging_keeps_its_own_line(worker):
    """Дрібна перевага сусіда не варта того, щоб ламати власну траєкторію."""
    worker.stagnation_counter = 0
    barely_better = (worker.run_best_cost * 0.999, [1] * worker.ctx.num_pipes)
    before = list(worker.pool.active_pool)

    worker._maybe_migrate(barely_better, {})

    assert worker.pool.active_pool == before


def test_no_global_best_yet_is_not_an_error(worker):
    worker._maybe_migrate(None, {"global_archive": [(1.0, [0, 0])]})

    assert worker.global_archive == [(1.0, [0, 0])], (
        "архів оновлюється навіть тоді, коли мігрувати нема до кого")


# --- три політики -----------------------------------------------------

def test_a_cold_island_takes_the_hybrid_into_the_pool_only(worker):
    worker.stagnation_counter = 0          # T = 0 -> холодна політика
    before_best = worker.run_best_cost
    before_size = len(worker.pool.active_pool)

    worker._maybe_migrate(global_best(worker, 0.5), {})

    assert len(worker.pool.active_pool) == before_size + 1
    assert worker.run_best_cost == before_best, (
        "холодна міграція не чіпає власний рекорд")
    assert worker.stagnation_counter == 0
    assert worker.ipc_immunity == 50, "після ін'єкції воркер тимчасово глухий до IPC"


def test_a_warm_island_may_adopt_the_hybrid_as_its_record(worker):
    worker.stag_limit, worker.stagnation_counter = 4, 6   # T = 0.5
    cheap = [0] * worker.ctx.num_pipes

    worker._maybe_migrate(global_best(worker, 0.3, cheap), {})

    # гібрид приймається лише якщо вийшов дешевшим — перевіряємо інваріант,
    # а не конкретне число, бо схрещування випадкове
    assert worker.run_best_cost <= worker.ctx.get_cached_stats(
        worker.run_best_sol).cost + 1e-9


def test_a_hot_island_ignores_its_neighbours(worker):
    worker.stag_limit, worker.stagnation_counter = 4, 100   # T = 1.0
    before_best = worker.run_best_cost
    before = list(worker.pool.active_pool)

    worker._maybe_migrate(global_best(worker, 0.5), {})

    assert worker.pool.active_pool == before
    assert worker.run_best_cost == before_best


# --- шум від сусіда ---------------------------------------------------

def test_a_peer_equal_to_the_global_best_is_not_used_as_noise(worker):
    """Сусід, який сам і є глобальним рекордом, шуму не додає."""
    gb_cost, gb_sol = global_best(worker, 0.5)
    progress = {"best_sol_1": (gb_cost, gb_sol), "best_sol_2": (gb_cost, gb_sol)}

    assert worker._pick_peer_solution(gb_cost, progress) is None


def test_a_distinct_peer_is_picked_as_noise(worker):
    gb_cost, _ = global_best(worker, 0.5)
    distinct = [3] * worker.ctx.num_pipes
    progress = {"best_sol_1": (gb_cost + 5.0, distinct)}

    assert worker._pick_peer_solution(gb_cost, progress) == distinct


def test_the_worker_never_picks_itself_as_a_peer(worker):
    gb_cost, _ = global_best(worker, 0.5)
    mine = [9] * worker.ctx.num_pipes
    progress = {"best_sol_0": (gb_cost + 5.0, mine)}

    assert worker._pick_peer_solution(gb_cost, progress) is None


def test_no_peers_reported_yet_is_not_an_error(worker):
    assert worker._pick_peer_solution(1.0, {}) is None


# --- архів донорів ----------------------------------------------------

def test_the_shared_archive_is_adopted_on_every_migration_window(worker):
    worker.stagnation_counter = 0
    archive = [(1.0, [0] * worker.ctx.num_pipes)]

    worker._maybe_migrate(global_best(worker, 0.5), {"global_archive": archive})

    assert worker.global_archive == archive
