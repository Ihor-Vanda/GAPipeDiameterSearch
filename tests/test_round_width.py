"""Ширина раунду: правило замість таблиці за класом мережі.

Скільки кандидатів на батька генерує раунд, досі залежало від класу
мережі (`CANDIDATE_DIVISOR`: 5 / 10 / 20 плюс окремий випадок для
XLARGE). Тепер ширина підлаштовується під **заміряну ціну раунду**:
з'їв понад `ROUND_BUDGET_SHARE` бюджету — звузити.

Перевіряється не число, а три властивості, кожна з яких може зламатись
непомітно:

* правило **одностороннє** — лише звужує. Розширення на малих мережах
  ніхто не міряв, і якщо регулятор колись почне розширювати, прогони на
  Ханої та Фоссі поїдуть, а еталон цього не спіймає одразу;
* воно **не опускається нижче підлоги** `MIN_CANDIDATES` — раунд без
  кандидатів це не дешевий раунд, а зупинений пошук;
* воно **справді викликається з головного циклу**. Скан тут навмисно
  самоохоронний: якщо виклик поїде з `run()`, лічильник лишиться нулем і
  тест впаде, а не почне мовчки проходити.
"""

import pytest

from pipeopt.solver.island import IslandWorker
from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch

GUARD_SECONDS = 20.0


def make_worker(ctx, max_sims=1000):
    local_search = LocalSearch(ctx)
    return IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                        worker_id=0, n_workers=1, max_sims=max_sims,
                        beam_width=5, global_archive=[])


@pytest.fixture
def worker(ctx):
    return make_worker(ctx)


# --- межа ціни раунду --------------------------------------------------

def test_the_limit_is_a_share_of_the_simulation_budget(worker):
    limit = worker.round_cost_limit(time_budget=3600.0)

    assert limit == pytest.approx(worker.max_sims * worker.ROUND_BUDGET_SHARE)


def test_without_a_simulation_budget_the_limit_comes_from_measured_speed(ctx):
    """Прогін за годинником: бюджет рахується зі **швидкості симулятора**.

    Взяти тут таблицю `SIM_BUDGET` за класом мережі означало б протягти
    назад саме те, що правило прибирає.
    """
    worker = make_worker(ctx, max_sims=float("inf"))

    limit = worker.round_cost_limit(time_budget=100.0)

    assert limit == pytest.approx(
        100.0 * ctx.sim_speed * worker.ROUND_BUDGET_SHARE)


# --- саме правило ------------------------------------------------------

def test_an_expensive_round_narrows_the_next_one(worker):
    worker._round_cost_limit = 100.0
    worker.SINGLE_CANDIDATES = 20

    narrowed = worker.narrow_round(spent=150.0)

    assert narrowed, "раунд вийшов дорожчим за межу — мав звузитись"
    assert worker.SINGLE_CANDIDATES < 20


def test_a_cheap_round_changes_nothing(worker):
    worker._round_cost_limit = 100.0
    worker.SINGLE_CANDIDATES = 20

    narrowed = worker.narrow_round(spent=40.0)

    assert not narrowed
    assert worker.SINGLE_CANDIDATES == 20, (
        "правило одностороннє: дешевий раунд не має права розширювати")


def test_the_rule_never_widens_even_after_many_cheap_rounds(worker):
    worker._round_cost_limit = 100.0
    start = worker.SINGLE_CANDIDATES

    for _ in range(50):
        worker.narrow_round(spent=1.0)

    assert worker.SINGLE_CANDIDATES == start


def test_narrowing_stops_at_the_floor(worker):
    floor = worker.ctx.profile.MIN_CANDIDATES
    worker._round_cost_limit = 1.0
    worker.SINGLE_CANDIDATES = 40

    for _ in range(50):
        worker.narrow_round(spent=1e6)

    assert worker.SINGLE_CANDIDATES == floor, (
        "раунд без кандидатів — це не дешевий раунд, а зупинений пошук")


def test_at_the_floor_the_rule_reports_no_change(worker):
    worker._round_cost_limit = 1.0
    worker.SINGLE_CANDIDATES = worker.ctx.profile.MIN_CANDIDATES

    assert not worker.narrow_round(spent=1e6)


# --- підключення до головного циклу ------------------------------------

def test_the_main_loop_measures_every_round(ctx, monkeypatch):
    """Правило має викликатись із циклу, інакше воно мертве.

    Тест самоохоронний: якщо виклик зникне з `run()`, `seen` лишиться
    порожнім і перевірка впаде.
    """
    worker = make_worker(ctx, max_sims=800)
    seen = []
    original = IslandWorker.narrow_round

    def spy(self, spent):
        seen.append(spent)
        return original(self, spent)

    monkeypatch.setattr(IslandWorker, "narrow_round", spy)

    worker.run(GUARD_SECONDS, float("inf"), None)

    assert seen, "правило жодного разу не викликалось із головного циклу"
    assert all(s >= 0 for s in seen), "ціна раунду не може бути від'ємною"


def test_a_cheap_network_keeps_its_round_width(ctx):
    """Мережа, де раунд дешевий, має пройти прогін без жодного звуження.

    Це і є причина, чому еталон Ханої лишається побітово тим самим.
    """
    worker = make_worker(ctx, max_sims=800)
    start = worker.SINGLE_CANDIDATES

    worker.run(GUARD_SECONDS, float("inf"), None)

    assert worker.SINGLE_CANDIDATES == start
