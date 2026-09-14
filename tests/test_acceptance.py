"""Правило прийняття результату кіка: пороги й класифікація.

Усі пороги залежать від температури, і саме тому в `_apply_kick` вони
були розсипані інлайновими виразами — `1.03 + 0.07 * T`, `1.5 + 0.5 * T`,
`0.20 if T > 0.6 else (0.08 if is_late_game else 0.15)`. Тут вони мають
імена, і кожен перевіряється на межі.

Окремо перевіряється, що `is_ghost` і `is_basin_tabu` викликаються
**тільки** в тих гілках, де вони потрібні: `is_ghost_solution` виконує
симуляції, тому зайвий виклик змінив би кількість симуляцій у прогоні.
"""

import pytest

from analytical_solver.acceptance import AcceptancePolicy, Verdict


@pytest.fixture
def policy():
    return AcceptancePolicy()


class Spy:
    """Функція-відповідь, яка рахує, скільки разів її спитали."""

    def __init__(self, answer=False):
        self.answer = answer
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.answer


def classify(policy, **over):
    args = dict(is_escape=False, effective_cost=100.0, deficit=0.0, best_cost=200.0,
                temperature=0.1, is_basin_tabu=Spy(False), is_ghost=Spy(False))
    args.update(over)
    return policy.classify(**args)


# --- пороги -----------------------------------------------------------

def test_explosion_threshold_widens_with_temperature(policy):
    assert policy.explosion_threshold(1000.0, 0.0) == pytest.approx(1500.0)
    assert policy.explosion_threshold(1000.0, 1.0) == pytest.approx(2000.0)


def test_pool_window_widens_with_temperature(policy):
    assert policy.pool_window(1000.0, 0.0) == pytest.approx(1030.0)
    assert policy.pool_window(1000.0, 1.0) == pytest.approx(1100.0)


def test_deficit_tolerance_is_proportional_to_temperature(policy):
    assert policy.deficit_tolerance(0.0) == 0.0
    assert policy.deficit_tolerance(1.0) == pytest.approx(0.5)


def test_relaxed_pressure_bar_drops_with_temperature(policy):
    assert policy.relaxed_h_min(30.0, 0.0) == 30.0
    assert policy.relaxed_h_min(30.0, 1.0) == pytest.approx(29.5)


def test_pressure_is_hopeless_only_beyond_both_limits(policy):
    """Порог — max(20 м, h_min × (1 + T)); при h_min = 30 і T = 0 це 30 м."""
    assert policy.is_hopeless_pressure(30.0, -1.0, 0.0) is True     # недобір 31 м
    assert policy.is_hopeless_pressure(30.0, 5.0, 0.0) is False     # недобір 25 м
    assert policy.is_hopeless_pressure(30.0, -1.0, 1.0) is False    # порог 60 м


def test_small_h_min_still_keeps_the_twenty_metre_floor(policy):
    """Без підлоги 20 м мережа з низькою вимогою тиску відкидала б усе."""
    assert policy.is_hopeless_pressure(1.0, -10.0, 0.0) is False    # недобір 11 м
    assert policy.is_hopeless_pressure(1.0, -25.0, 0.0) is True     # недобір 26 м


def test_deficit_is_converted_into_cost(policy):
    # 10 % рекорду за кожен метр недобору
    assert policy.effective_cost(1000.0, 0.0, 5000.0) == 1000.0
    assert policy.effective_cost(1000.0, 2.0, 5000.0) == pytest.approx(2000.0)


def test_heartbeat_margin_narrows_as_the_run_progresses(policy):
    assert policy.heartbeat_margin(0.0) == pytest.approx(0.03)
    assert policy.heartbeat_margin(1.0) == pytest.approx(0.01)


def test_large_network_pool_gap_depends_on_phase(policy):
    assert policy.max_pool_gap(0.9, is_late_game=False) == 0.20
    assert policy.max_pool_gap(0.1, is_late_game=True) == 0.08
    assert policy.max_pool_gap(0.1, is_late_game=False) == 0.15


# --- глибина поліру ---------------------------------------------------

def test_hot_solutions_get_the_shallowest_polish(policy):
    assert policy.deep_passes(gap=-0.5, temperature=0.7, num_pipes=34) == 1
    assert policy.deep_passes(gap=-0.5, temperature=0.5, num_pipes=34) == 2


def test_a_cold_improvement_earns_the_deepest_polish(policy):
    assert policy.deep_passes(gap=-0.01, temperature=0.1, num_pipes=34) == 5
    assert policy.deep_passes(gap=-0.01, temperature=0.1, num_pipes=454) == 8


def test_polish_depth_falls_off_as_the_gap_grows(policy):
    assert policy.deep_passes(gap=0.001, temperature=0.1, num_pipes=34) == 3
    assert policy.deep_passes(gap=0.01, temperature=0.1, num_pipes=34) == 2
    assert policy.deep_passes(gap=0.04, temperature=0.1, num_pipes=34) == 2
    assert policy.deep_passes(gap=0.10, temperature=0.1, num_pipes=34) == 1


# --- класифікація -----------------------------------------------------

def test_a_cheaper_feasible_solution_is_a_record(policy):
    assert classify(policy, effective_cost=100.0, best_cost=200.0) == Verdict.RECORD


def test_a_cheaper_solution_with_a_deficit_is_not_a_record(policy):
    """Недобір тиску знімає право на запис, навіть якщо вартість менша."""
    assert classify(policy, effective_cost=100.0, best_cost=200.0,
                    deficit=0.2) != Verdict.RECORD


def test_a_suspiciously_large_gain_is_checked_for_a_simulation_artefact(policy):
    ghost = Spy(True)
    verdict = classify(policy, effective_cost=100.0, best_cost=200.0, is_ghost=ghost)

    assert verdict == Verdict.GHOST_VETO
    assert ghost.calls == 1


def test_a_small_gain_is_never_checked_for_an_artefact(policy):
    """Перевірка виконує симуляції, тому для малого виграшу її не роблять."""
    ghost = Spy(True)
    verdict = classify(policy, effective_cost=199.0, best_cost=200.0, is_ghost=ghost)

    assert verdict == Verdict.RECORD
    assert ghost.calls == 0


def test_escape_is_adopted_even_though_it_is_not_cheaper(policy):
    assert classify(policy, is_escape=True, effective_cost=300.0,
                    best_cost=200.0) == Verdict.ESCAPE


def test_escape_with_a_deficit_is_not_adopted(policy):
    assert classify(policy, is_escape=True, effective_cost=300.0, best_cost=200.0,
                    deficit=0.1) != Verdict.ESCAPE


def test_a_solution_inside_the_window_goes_to_the_pool(policy):
    verdict = classify(policy, effective_cost=205.0, best_cost=200.0)
    assert verdict == Verdict.POOL
    assert verdict.gap == pytest.approx(0.025)


def test_a_tabu_basin_is_not_accepted_into_the_pool(policy):
    assert classify(policy, effective_cost=205.0, best_cost=200.0,
                    is_basin_tabu=Spy(True)) == Verdict.REJECT


def test_pool_window_is_always_tighter_than_the_gap_limit(policy):
    """Фільтр розриву для великих мереж недосяжний — і мусить лишатись таким,
    доки вікно не розширять.

    Задум був такий: рішення, що пройшло у вікно пулу, але на великій
    мережі дороге за рекорд більш ніж на `max_pool_gap`, у пул не
    потрапляє. Але вікно — `0.03 + 0.07·T`, максимум 0.10 при T = 1, а
    найменша межа розриву — 0.08. Тобто вікно завжди тугіше, і рішення,
    яке його пройшло, ніколи не перевищує межу. Гілка була мертвою.

    Цей тест впаде, якщо хтось розширить вікно: тоді фільтр треба
    повертати свідомо (§20.8), а не отримати його раптом робочим.
    """
    for step in range(101):
        temperature = step / 100
        window_gap = (policy.POOL_WINDOW_BASE
                      + policy.POOL_WINDOW_TEMPERATURE_GAIN * temperature) - 1.0
        for is_late_game in (False, True):
            assert window_gap <= policy.max_pool_gap(temperature, is_late_game)


def test_a_solution_inside_the_window_goes_to_the_pool_on_any_network_size(policy):
    assert classify(policy, effective_cost=205.0, best_cost=200.0) == Verdict.POOL


def test_hail_mary_only_at_the_highest_temperature(policy):
    assert classify(policy, effective_cost=500.0, best_cost=200.0,
                    temperature=1.0) == Verdict.HAIL_MARY
    assert classify(policy, effective_cost=500.0, best_cost=200.0,
                    temperature=0.9) == Verdict.REJECT


def test_hail_mary_also_refuses_a_tabu_basin(policy):
    assert classify(policy, effective_cost=500.0, best_cost=200.0, temperature=1.0,
                    is_basin_tabu=Spy(True)) == Verdict.REJECT


def test_a_record_never_asks_about_the_basin(policy):
    """У рекорді табу басейну не перевіряється — так було в оригіналі."""
    tabu = Spy(True)
    verdict = classify(policy, effective_cost=100.0, best_cost=200.0, is_basin_tabu=tabu)

    assert verdict == Verdict.RECORD
    assert tabu.calls == 0


def test_everything_else_is_rejected(policy):
    assert classify(policy, effective_cost=500.0, best_cost=200.0,
                    temperature=0.1) == Verdict.REJECT


# --- ablation ---------------------------------------------------------

def test_ablation_freezes_the_temperature():
    assert AcceptancePolicy(ablation=True).effective_temperature(0.9) == 0.5
    assert AcceptancePolicy().effective_temperature(0.9) == 0.9


def test_ablation_removes_the_pressure_tolerance():
    ablated = AcceptancePolicy(ablation=True)
    assert ablated.deficit_tolerance(1.0) == 0.0
    assert ablated.relaxed_h_min(30.0, 1.0) == 30.0
