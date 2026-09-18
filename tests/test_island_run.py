"""Головний цикл острова: від зерен до підсумкового рішення.

`IslandWorker.run` — це те, заради чого існує решта `island.py`, і воно
лишалось найменш перевіреним. Не через складність: через **вартість
перевірки**. Дістатись сюди прогоном означає підняти EPANET, а на
справжній мережі один оберт циклу — це тисячі симуляцій.

Але `run` не знає, що симулятор справжній. Із `FakeSimulator` (ланцюг з
аналітичною формулою втрат) увесь цикл виконується за мілісекунди —
разом із міжострівним шаром, аварійним відновленням, підсумковим записом
у `shared_progress` і фінальним щитом.

Що саме тут важливо перевірити:

* **бюджет дотримується**. Зупинка по симуляціях і по годиннику — дві
  різні гілки одного `if`, і помилка в будь-якій означає прогін, який не
  зупиняється, або зупиняється одразу. На восьмигодинному прогоні це
  помітно не одразу;
* **підсумок доходить до оркестратора**. Останній запис у
  `shared_progress` робиться **після** виходу з циклу. Якщо він не
  відбувся, головний процес побачить передостанній стан острова —
  занижений лічильник симуляцій і, можливо, не той рекорд;
* **фінальний щит**. Останнє, що робить острів, — перевіряє, що його
  рішення справді тримає тиск, і лікує, якщо ні. Без цього недопустиме
  рішення поїде в звіт із позначкою «OK».
"""

import random

import pytest

from pipeopt.solver.island import IslandWorker
from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch


#: Годинниковий запобіжник для наскрізних прогонів. Зупиняє їх не він —
#: зупиняє бюджет симуляцій; але без нього помилка в умові виходу
#: означала б тест, який висить, а не падає.
GUARD_SECONDS = 20.0


def make_worker(ctx, max_sims=1000, n_workers=1, worker_id=0):
    """Острів на ланцюговій мережі з мікробюджетом.

    Бюджет саме мікроскопічний, і причина не в швидкості. У мережі з
    шести труб і чотирьох діаметрів усього 4096 різних рішень, тому
    після перших сотень симуляцій кеш поглинає майже все: пошук
    продовжує працювати, а лічильник симуляцій майже не росте. Бюджет у
    кілька тисяч на такій мережі вичерпується хвилинами — не через
    складність, а через те, що зупинка міряється симуляціями, яких уже
    не відбувається. На справжній мережі цього не буває.
    """
    local_search = LocalSearch(ctx)
    return IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                        worker_id=worker_id, n_workers=n_workers,
                        max_sims=max_sims, beam_width=5, global_archive=[])


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(2026)


@pytest.fixture
def worker(ctx):
    return make_worker(ctx)


# --- прогін цілком ------------------------------------------------------

def test_a_run_returns_a_feasible_solution(worker, ctx):
    """Найпростіше твердження — і найважливіше: рішення тримає тиск."""
    cost, solution = worker.run(GUARD_SECONDS, float("inf"), None)

    assert solution is not None
    assert len(solution) == ctx.num_pipes
    assert ctx.get_cached_stats(solution)[1] >= ctx.h_min
    assert cost == pytest.approx(ctx.get_cached_stats(solution)[0])


def test_the_simulation_budget_is_respected(ctx):
    """Бюджет — єдине, що зупиняє пошук у режимі без годинника.

    Перевіряється з запасом, а не точною рівністю: один оберт циклу
    витрачає кілька симуляцій, тому перевищення на величину одного
    оберту нормальне, а вдвічі — ні.
    """
    worker = make_worker(ctx, max_sims=800)

    worker.run(GUARD_SECONDS, float("inf"), None)

    assert worker.ctx.sim_count < 800 * 2


def test_a_zero_time_budget_stops_the_run_at_once(worker, ctx):
    """Годинник — друга гілка тієї самої зупинки.

    Нульовий бюджет означає «часу немає»: острів мусить повернути те, що
    дали зерна, а не крутити цикл.
    """
    cost, solution = worker.run(0.0, float("inf"), None)

    assert solution is not None
    assert cost < float("inf"), "навіть без жодного раунду зерна дають рішення"


def test_a_cheaper_run_beats_the_seeds(worker, ctx):
    """Сенс циклу: підсумок мусить бути не гіршим за найкраще зерно."""
    worker.run(GUARD_SECONDS, float("inf"), None)
    from_seeds = min(
        ctx.get_cached_stats(seed)[0] for seed in worker.seeder.make_diverse_seeds())

    assert worker.run_best_cost <= from_seeds


# --- звіт оркестраторові ------------------------------------------------

def test_the_final_state_reaches_the_orchestrator(ctx):
    """Останній запис робиться **після** виходу з циклу.

    Без нього головний процес бачить передостанній стан острова:
    занижений лічильник симуляцій і, можливо, не той рекорд. У логах це
    не видно ніяк — просто числа трохи інші.
    """
    worker = make_worker(ctx, max_sims=1200)
    shared = {0: 0}

    worker.run(GUARD_SECONDS, float("inf"), shared)

    report = shared[0]
    assert isinstance(report, dict)
    assert report["sims"] == worker.ctx.sim_count
    assert report["best_cost"] == worker.run_best_cost


def test_a_broken_shared_dictionary_does_not_kill_the_island(ctx):
    """`Manager().dict()` живе в іншому процесі — він може відвалитись.

    Втратити зв'язок прикро, але це не причина втрачати результат
    багатогодинного пошуку: острів дописує невдачу в лог і повертає
    рішення.
    """
    class Disconnected(dict):
        def __setitem__(self, key, value):
            raise BrokenPipeError("менеджер недоступний")

    worker = make_worker(ctx, max_sims=800)

    cost, solution = worker.run(GUARD_SECONDS, float("inf"), Disconnected())

    assert solution is not None
    assert cost < float("inf")


# --- фінальний щит ------------------------------------------------------

def test_an_invalid_final_solution_is_healed(worker, ctx, capsys):
    """Останнє, що робить острів, — перевіряє власний результат.

    Рішення могло стати недопустимим через артефакт кешу симуляцій
    (§«щит від ілюзій»). Без цієї перевірки воно поїхало б у звіт із
    позначкою «OK» — найгірший різновид помилки, бо результат виглядає
    правильним.
    """
    worker.run(0.0, float("inf"), None)
    worker.run_best_sol = [0] * ctx.num_pipes
    worker.run_best_cost = ctx.get_cached_stats(worker.run_best_sol)[0]
    assert ctx.get_cached_stats(worker.run_best_sol)[1] < ctx.h_min

    cost, solution = worker.run(0.0, float("inf"), None)

    assert ctx.get_cached_stats(solution)[1] >= ctx.h_min
    assert cost < float("inf")


# --- діагностика тісноти обмеження --------------------------------------

def test_the_run_reports_how_tight_the_pressure_constraint_is(worker, capsys):
    """Замір, який замінив прибране вікно відносної допустимості.

    Величина нічого не вирішує — вона показує, з якою мережею має справу
    солвер: чи обмеження тисне на кожен хід, чи майже не помітне. Замір
    безкоштовний, недобір уже обчислений у `evaluate_candidate`.
    """
    worker.run(GUARD_SECONDS, float("inf"), None)

    assert "[TIGHTNESS]" in capsys.readouterr().out


# --- ініціалізація зерен ------------------------------------------------

def test_the_best_feasible_seed_becomes_the_baseline(worker, ctx):
    """Стартовий рекорд — найдешевше **допустиме** зерно, а не найдешевше."""
    cheap_but_broken = [0] * ctx.num_pipes
    feasible = [2] * ctx.num_pipes
    worker.base_dyn_bonus = 0.0

    worker._initialize_seeds([cheap_but_broken, feasible])

    assert worker.run_best_sol == feasible
    assert ctx.get_cached_stats(cheap_but_broken)[0] < worker.run_best_cost, (
        "недопустиме зерно дешевше — саме тому його не можна брати")


def test_every_seed_goes_into_the_pool_and_the_tabu(worker, ctx):
    """Пул — матеріал для кіків; табу не дає повернутись у ті самі точки."""
    seeds = [[1] * ctx.num_pipes, [2] * ctx.num_pipes, [3] * ctx.num_pipes]
    worker.base_dyn_bonus = 0.0

    worker._initialize_seeds(seeds)

    assert len(worker.pool.active_pool) == len(seeds)
    assert worker.pool.tabu_fingerprints


def test_a_run_without_seeds_falls_back_to_the_widest_pipes(worker, ctx, capsys):
    """Гілка, яку прогін не показує: порожній список зерен.

    Порожній пул означав би падіння на першому ж кіку. Найбільші
    діаметри — завідомо допустиме рішення, від якого можна почати
    здешевлювати.
    """
    worker.base_dyn_bonus = 0.0

    worker._initialize_seeds([])

    assert worker.run_best_sol == [ctx.max_d_idx] * ctx.num_pipes
    assert "No valid seeds" in capsys.readouterr().out


def test_when_no_seed_holds_the_pressure_the_cheapest_one_starts_the_search(
        worker, ctx):
    """Третя гілка: зерна є, але жодне не допустиме.

    Повертати `None` не можна — пошуку потрібна точка старту. Береться
    найдешевше; фінальний щит наприкінці прогону відповідає за те, щоб
    недопустиме рішення не доїхало до звіту.
    """
    worker.base_dyn_bonus = 0.0
    too_narrow = [[0] * ctx.num_pipes, [0] * (ctx.num_pipes - 1) + [1]]

    worker._initialize_seeds(too_narrow)

    assert worker.run_best_sol is not None
    assert worker.run_best_cost == min(
        ctx.get_cached_stats(seed)[0] for seed in too_narrow)
