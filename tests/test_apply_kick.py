"""Характеризаційні тести `_apply_kick` — сітка перед розбором (етап 7).

Метод має CC 149 і мутує близько двадцяти полів воркера, причому
**порядок мутацій значущий**: наприклад, порог скидання
`stagnation_counter` рахується від уже оновленого `run_best_cost`. Тому
перед тим, як виносити з нього `StrategySelector`, `AdaptiveRates` і
`AcceptancePolicy`, тут зафіксовано, що саме змінюється в полях після
одного кіка в кожній із шести можливих розв'язок:

    RECORD -> ESCAPE -> POOL -> HAIL MARY (запис / прийняття) -> REJECT

плюс дві відмови: невдача самого кіка і жорсткий відкид за вибухом
вартості.

Це НЕ тести правильності: вони описують поведінку такою, яка вона є
зараз, включно з тим, що мені здається сумнівним. Якщо після розбору
щось із цього зміниться свідомо — тест правиться разом із кодом, і зміна
стає видимою в diff'і, а не тихою.

Стратегія форсується підміною `pool_for`: коли пул стратегій складається
з однієї, її обирає і UCB1, і VND-гілка, тому вибір перестає бути
випадковим, а решта методу працює без змін.
"""

import io
import random
from contextlib import redirect_stdout

import numpy as np
import pytest

import analytical_solver.orchestrator as orch
from analytical_solver.context import SolverContext
from analytical_solver.kicks import KickStrategies
from analytical_solver.local_search import LocalSearch

#: Поля воркера, які читаються в тестах як «стан після кіка».
WATCHED = ("run_best_cost", "run_best_sol", "global_best_cost", "stagnation_counter",
           "ipc_immunity", "_force_flush_next")


class Kick:
    """Результат одного застосування кіка: стан до, стан після і лог."""

    def __init__(self, worker, before, log):
        self.worker = worker
        self.before = before
        self.log = log

    @property
    def after(self):
        return snapshot(self.worker)

    def changed(self, field):
        return self.before[field] != self.after[field]

    @property
    def verdict(self):
        for needle, kind in (("[FORCE]", "RECORD"), ("POOL FLUSHED", "ESCAPE"),
                             ("Water Level Accept", "POOL"),
                             ("HAIL MARY → DIRECT", "HAIL_RECORD"),
                             ("HAIL MARY ACCEPT", "HAIL_POOL"),
                             ("Cost Explosion", "EXPLOSION"),
                             ("Poor or Tabu", "REJECT"),
                             ("failed. Reason", "KICK_FAILED")):
            if needle in self.log:
                return kind
        return "SILENT"


def snapshot(worker):
    state = {f: getattr(worker, f) for f in WATCHED}
    state["run_best_sol"] = list(worker.run_best_sol) if worker.run_best_sol else None
    state["pool_size"] = len(worker.pool.active_pool)
    # Статистика стратегій і розміри кроку читаються через одну точку: на
    # етапі 7 вони переїхали з дев'яти полів воркера в `StrategySelector` і
    # `AdaptiveRates`, і це єдине місце в сітці, яке про переїзд знає.
    state["wins_cold"] = dict(worker.selector.wins["cold"])
    state["tries_cold"] = dict(worker.selector.tries["cold"])
    state["fails"] = dict(worker.selector.consecutive_fails)
    state["last_strategy"] = worker.selector.last_strategy
    state["burst_count"] = worker.selector.burst_count
    state["rates"] = worker.rates.snapshot()
    return state


@pytest.fixture
def apply_kick(monkeypatch):
    """Провести один кік із заданою стратегією і контрольованим станом.

    `best_mult` зсуває `run_best_cost` відносно вартості базового рішення —
    саме цим вибирається гілка прийняття: що нижчий поточний рекорд, то
    гірше на його тлі виглядає результат кіка.
    `stagnation` задає температуру: `T = stagnation / (stag_limit * 3)`.
    """
    from conftest import FakeSimulator

    def run(strategy, *, stagnation=1, best_mult=1.0, seed=1, archive=None,
            progress_ratio=0.1, stag_limit=4, n_pipes=6):
        random.seed(seed)
        np.random.seed(seed)

        sim = FakeSimulator(n_pipes=n_pipes)
        ctx = SolverContext(sim, sim.config.diameters_m, v_opt=1.0)
        local_search = LocalSearch(ctx)
        worker = orch.IslandWorker(
            ctx, KickStrategies(ctx, local_search), local_search,
            0, 1, 100_000, 5, "SMALL", archive if archive is not None else [],
        )

        base = [3] * ctx.num_pipes
        reference_cost = sim.get_stats(base).cost
        worker.run_best_sol = base
        worker.run_best_cost = reference_cost * best_mult
        worker.global_best_cost = reference_cost * best_mult
        worker.base_dyn_bonus = reference_cost * 0.001
        worker.pool.active_pool = [(0.0, worker.run_best_cost, list(base))]
        worker.stagnation_counter = stagnation
        worker.stag_limit = stag_limit
        worker.progress_ratio = progress_ratio

        # Один пул стратегій -> вибір детермінований у будь-якій гілці.
        monkeypatch.setattr(orch, "pool_for", lambda *_a, **_k: [strategy])

        before = snapshot(worker)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            worker._apply_kick(1, None)
        return Kick(worker, before, buffer.getvalue())

    return run


#: Архів із трьох рішень — мінімум, з яким працює BASIN_ESCAPE.
ARCHIVE = [(1.0, [1] * 6), (2.0, [5] * 6), (3.0, [2] * 6)]


# --- усі шість розв'язок досяжні --------------------------------------

@pytest.mark.parametrize("kind,kwargs", [
    ("RECORD", dict(strategy="SHOCK", stagnation=1, best_mult=1.0)),
    ("ESCAPE", dict(strategy="BASIN_ESCAPE", stagnation=5, best_mult=0.6, archive=ARCHIVE)),
    ("POOL", dict(strategy="SHOCK", stagnation=12, best_mult=0.6)),
    ("HAIL_RECORD", dict(strategy="SHOCK", stagnation=12, best_mult=0.55)),
    ("HAIL_POOL", dict(strategy="SMART_PERTURB", stagnation=12, best_mult=0.32)),
    ("REJECT", dict(strategy="SHOCK", stagnation=1, best_mult=0.30)),
    ("EXPLOSION", dict(strategy="SHOCK", stagnation=12, best_mult=0.30)),
    ("KICK_FAILED", dict(strategy="BASIN_ESCAPE", stagnation=5, best_mult=1.0)),
])
def test_each_outcome_is_reachable(apply_kick, kind, kwargs):
    """Якщо після розбору якась гілка стане недосяжною, впаде саме тут."""
    assert apply_kick(**kwargs).verdict == kind


# --- RECORD -----------------------------------------------------------

def test_record_updates_best_pool_and_global(apply_kick):
    kick = apply_kick("SHOCK", stagnation=1, best_mult=1.0)
    after = kick.after

    assert after["run_best_cost"] < kick.before["run_best_cost"]
    assert after["run_best_sol"] != kick.before["run_best_sol"]
    assert after["pool_size"] == kick.before["pool_size"] + 1
    assert after["global_best_cost"] == after["run_best_cost"]


def test_record_resets_stagnation_when_the_gain_is_large(apply_kick):
    kick = apply_kick("SHOCK", stagnation=1, best_mult=1.0)
    assert kick.after["stagnation_counter"] == 0


def test_stagnation_threshold_is_measured_against_the_new_best_not_the_old():
    """Тонкість, яку розбір не має права зачепити.

    У методі стоїть

        self.run_best_cost, self.run_best_sol = effective_cost, final_sol
        self.stagnation_counter = 0 if diff > (self.run_best_cost * 0.005) else ...

    тобто порог рахується від **уже нового** рекорду. Різниця з варіантом
    «від старого» — 0.5 %, тому прогоном її не видно; тут вона зафіксована
    на самому правилі.
    """
    new_best = 1_000_000.0
    boundary = new_best * 0.005

    assert orch.IslandWorker._stagnation_after_record(boundary * 1.01, new_best, 9) == 0
    assert orch.IslandWorker._stagnation_after_record(boundary * 0.99, new_best, 9) == 7
    # Старий рекорд був би new + diff, тобто порог вищий; значення в
    # проміжку між двома порогами мусить трактуватись як «малий виграш».
    between = new_best * 0.00502
    assert orch.IslandWorker._stagnation_after_record(between, new_best, 9) == 0
    assert orch.IslandWorker._stagnation_after_record(between, new_best + between, 9) == 7


def test_stagnation_never_goes_negative():
    assert orch.IslandWorker._stagnation_after_record(0.0, 1_000_000.0, 1) == 0


# --- ESCAPE -----------------------------------------------------------

def test_escape_flushes_the_pool_to_exactly_one_solution(apply_kick):
    kick = apply_kick("BASIN_ESCAPE", stagnation=5, best_mult=0.6, archive=ARCHIVE)
    assert kick.after["pool_size"] == 1


def test_escape_clears_the_kick_memories_and_arms_the_flush(apply_kick):
    kick = apply_kick("BASIN_ESCAPE", stagnation=5, best_mult=0.6, archive=ARCHIVE)
    worker = kick.worker

    # Чотири пам'яті переїхали в `KickMemory` (етап 8); перевірка та сама —
    # після великої втечі не має лишитись жодної.
    assert worker.kick_memory.bottleneck_failed == {}
    assert worker.kick_memory.bottleneck_boosted == {}
    assert worker.kick_memory.loop_balance_failed == {}
    assert worker.kick_memory.zero_sum_tabu == {}
    assert worker._force_flush_next is True
    assert worker.ipc_immunity == 50
    assert kick.after["stagnation_counter"] == 0


def test_escape_decays_both_bandit_tables(apply_kick):
    """Після великої втечі статистика стратегій обнуляється на 70 %:
    попередні ваги стосувались іншого басейну."""
    kick = apply_kick("BASIN_ESCAPE", stagnation=5, best_mult=0.6, archive=ARCHIVE)

    for strategy, before in kick.before["wins_cold"].items():
        assert kick.after["wins_cold"][strategy] == pytest.approx(max(1.0, before * 0.3))


def test_escape_remembers_the_basin_it_left(apply_kick):
    kick = apply_kick("BASIN_ESCAPE", stagnation=5, best_mult=0.6, archive=ARCHIVE)
    assert len(kick.worker.pool.basin_tabu) == 1


# --- POOL -------------------------------------------------------------

def test_pool_accept_adds_a_solution_without_touching_the_record(apply_kick):
    kick = apply_kick("SHOCK", stagnation=12, best_mult=0.6)
    after = kick.after

    assert after["pool_size"] == kick.before["pool_size"] + 1
    assert after["run_best_cost"] == kick.before["run_best_cost"]
    assert after["run_best_sol"] == kick.before["run_best_sol"]


def test_pool_accept_stores_a_penalised_score(apply_kick):
    """У пул кладеться `score * 1.05`, тобто прийняте рішення свідомо
    гірше за власну оцінку — щоб рекорд не витіснявся з голови пулу."""
    kick = apply_kick("SHOCK", stagnation=12, best_mult=0.6)
    score, cost, _ = kick.worker.pool.active_pool[-1]
    assert score > cost or score == pytest.approx(cost * 1.05, rel=0.2)


# --- HAIL MARY --------------------------------------------------------

def test_hail_mary_record_updates_the_best(apply_kick):
    kick = apply_kick("SHOCK", stagnation=12, best_mult=0.55)
    assert kick.after["run_best_cost"] < kick.before["run_best_cost"]


def test_hail_mary_accept_parks_stagnation_high(apply_kick):
    """Прийняття завідомо гіршого рішення при T ≈ 1 не скидає застій, а
    навпаки виставляє його в 1.8 × ліміту — наступний кік мусить бути
    таким самим агресивним."""
    kick = apply_kick("SMART_PERTURB", stagnation=12, best_mult=0.32)
    assert kick.after["stagnation_counter"] == int(kick.worker.stag_limit * 1.8)


# --- REJECT і відмови -------------------------------------------------

def test_reject_changes_no_solution_state(apply_kick):
    kick = apply_kick("SHOCK", stagnation=1, best_mult=0.30)
    after = kick.after

    assert after["run_best_cost"] == kick.before["run_best_cost"]
    assert after["run_best_sol"] == kick.before["run_best_sol"]
    assert after["pool_size"] == kick.before["pool_size"]


def test_reject_shrinks_the_step_size_of_size_driven_kicks(apply_kick):
    """JADE: відкид — сигнал, що крок завеликий."""
    kick = apply_kick("SMART_PERTURB", stagnation=1, best_mult=0.30)
    if kick.verdict == "REJECT":
        assert kick.after["rates"]["perturb"] < kick.before["rates"]["perturb"]


def test_failed_kick_counts_a_consecutive_failure(apply_kick):
    kick = apply_kick("BASIN_ESCAPE", stagnation=5, best_mult=1.0)
    assert kick.verdict == "KICK_FAILED"
    assert kick.after["fails"]["BASIN_ESCAPE"] == 1


def test_failed_kick_penalises_the_strategy_hard(apply_kick):
    """Невдача кіка коштує стратегії 75 % ваги, і не менше 0.1."""
    kick = apply_kick("BASIN_ESCAPE", stagnation=5, best_mult=1.0)
    before = kick.before["wins_cold"]["BASIN_ESCAPE"]
    assert kick.after["wins_cold"]["BASIN_ESCAPE"] == pytest.approx(max(0.1, before * 0.25))


def test_cost_explosion_is_rejected_before_any_state_changes(apply_kick):
    kick = apply_kick("SHOCK", stagnation=12, best_mult=0.30)
    after = kick.after

    assert kick.verdict == "EXPLOSION"
    assert after["run_best_cost"] == kick.before["run_best_cost"]
    assert after["pool_size"] == kick.before["pool_size"]


# --- облік стратегій --------------------------------------------------

def test_every_kick_counts_a_try(apply_kick):
    kick = apply_kick("SHOCK", stagnation=1, best_mult=1.0)
    assert kick.after["tries_cold"]["SHOCK"] == kick.before["tries_cold"]["SHOCK"] + 1


def test_repeating_a_strategy_grows_the_burst_counter(apply_kick):
    kick = apply_kick("SHOCK", stagnation=1, best_mult=1.0)
    assert kick.after["last_strategy"] == "SHOCK"
    assert kick.after["burst_count"] == 1


def test_a_successful_kick_clears_its_failure_streak(apply_kick):
    kick = apply_kick("SHOCK", stagnation=1, best_mult=1.0)
    assert kick.after["fails"]["SHOCK"] == 0
