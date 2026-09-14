"""Вибір стратегії кіка: два режими, дві таблиці, сім правил навчання.

Найважливіше — що «гаряча» й «холодна» таблиці справді незалежні: у
другій половині прогону бандит вчиться з чистого аркуша, і все, що він
вивчив на початку, туди не переноситься. У початковому коді це
забезпечувалось парою псевдонімів `active_wins` / `active_tries`, і
переплутати таблицю було легко.
"""

import random

import pytest

from analytical_solver.strategy_selector import StrategySelector

STRATEGIES = ("SHOCK", "BOTTLENECK", "TRIM", "RUIN_RECREATE")


@pytest.fixture
def selector():
    random.seed(0)
    return StrategySelector(STRATEGIES)


def cold(selector, temperature=0.1):
    """Вибір у холодній фазі (перша половина прогону)."""
    return selector.pick(list(STRATEGIES), temperature, temperature, is_late_game=False)


def hot(selector, temperature=0.1):
    return selector.pick(list(STRATEGIES), temperature, temperature, is_late_game=True)


# --- режими -----------------------------------------------------------

def test_bandit_runs_while_the_run_is_cold_and_stagnation_is_moderate(selector):
    assert cold(selector, temperature=0.1).bandit_active is True


def test_high_stagnation_switches_to_random_variable_neighbourhood(selector):
    assert cold(selector, temperature=0.9).bandit_active is False


def test_late_game_switches_to_random_even_at_low_stagnation(selector):
    assert hot(selector, temperature=0.1).bandit_active is False


def test_ablation_mode_never_uses_the_bandit():
    random.seed(0)
    selector = StrategySelector(STRATEGIES, ablation=True)
    assert cold(selector, temperature=0.1).bandit_active is False


def test_the_label_says_which_mode_ran(selector):
    assert cold(selector, temperature=0.25).mode_label == "[UCB1 Temp: 0.25]"
    assert cold(selector, temperature=0.9).mode_label == "[VND RAND]"


# --- дві незалежні таблиці --------------------------------------------

def test_cold_and_hot_statistics_do_not_leak_into_each_other(selector):
    cold(selector)
    selector.reward("SHOCK", 2.0)

    assert selector.wins["cold"]["SHOCK"] > 1.0
    assert selector.wins["hot"]["SHOCK"] == 1.0


def test_rewards_land_in_the_table_of_the_last_pick(selector):
    hot(selector)
    selector.reward("SHOCK", 2.0)

    assert selector.wins["hot"]["SHOCK"] > 1.0
    assert selector.wins["cold"]["SHOCK"] == 1.0


def test_a_try_is_counted_in_the_active_table_only(selector):
    before_hot = selector.tries["hot"]["SHOCK"]
    choice = cold(selector)

    assert selector.tries["cold"][choice.strategy] > 1.0
    assert selector.tries["hot"]["SHOCK"] == before_hot


# --- UCB1 -------------------------------------------------------------

def test_the_bandit_prefers_the_strategy_that_has_been_winning(selector):
    cold(selector)
    # Однакова кількість спроб у всіх -> бонус за недовивченість однаковий,
    # і вибір вирішує тільки середній виграш.
    for strategy in STRATEGIES:
        selector.tries["cold"][strategy] = 20.0
    selector.wins["cold"]["TRIM"] = 4.0

    selector.last_strategy = None
    assert cold(selector).strategy == "TRIM"


def test_the_bandit_explores_a_strategy_it_has_barely_tried(selector):
    """Бонус за недовивченість мусить перебивати кращий середній виграш:
    стратегія з однією спробою цікавіша за перевірену, доки статистики
    на неї майже немає."""
    cold(selector)
    for strategy in STRATEGIES:
        selector.tries["cold"][strategy] = 20.0
    selector.wins["cold"]["TRIM"] = 4.0
    selector.tries["cold"]["BOTTLENECK"] = 1.0

    selector.last_strategy = None
    assert cold(selector).strategy == "BOTTLENECK"


def test_repeating_a_strategy_is_penalised_geometrically(selector):
    """Штраф 0.5^серія — саме він не дає бандиту залипнути."""
    selector.last_strategy = "TRIM"
    selector.burst_count = 4
    selector.wins["cold"]["TRIM"] = 4.0
    for _ in range(50):
        selector.tries["cold"]["TRIM"] += 1

    assert cold(selector).strategy != "TRIM"


def test_an_untried_strategy_is_always_taken_first():
    """Поки є стратегія без статистики, бандит не рахує нічого."""
    random.seed(0)
    selector = StrategySelector(("SHOCK",))
    choice = selector.pick(["SHOCK", "BRAND_NEW"], 0.1, 0.1, is_late_game=False)
    assert choice.strategy == "BRAND_NEW"


def test_burst_counter_grows_on_repeat_and_resets_on_change(selector):
    selector.pick(["SHOCK"], 0.1, 0.1, is_late_game=False)
    assert (selector.last_strategy, selector.burst_count) == ("SHOCK", 1)

    selector.pick(["SHOCK"], 0.1, 0.1, is_late_game=False)
    assert selector.burst_count == 2

    selector.pick(["TRIM"], 0.1, 0.1, is_late_game=False)
    assert (selector.last_strategy, selector.burst_count) == ("TRIM", 1)


# --- виведення з обігу за невдачами -----------------------------------

def test_a_strategy_with_too_many_failures_is_skipped(selector):
    for _ in range(StrategySelector.MAX_FAILS_COLD):
        selector.note_failure("SHOCK")

    picks = {selector.pick(["SHOCK", "TRIM"], 0.1, 0.1, is_late_game=False).strategy
             for _ in range(10)}
    assert picks == {"TRIM"}


def test_high_temperature_tolerates_more_failures(selector):
    """При сильному застої невдачі очікувані, тому поріг вищий."""
    for _ in range(StrategySelector.MAX_FAILS_COLD):
        selector.note_failure("SHOCK")

    picks = {selector.pick(["SHOCK"], 0.9, 0.9, is_late_game=False).strategy
             for _ in range(5)}
    assert picks == {"SHOCK"}


def test_all_strategies_exhausted_means_the_filter_is_ignored(selector):
    """Якщо всі вибули, вибір усе одно мусить щось повернути."""
    for strategy in STRATEGIES:
        for _ in range(10):
            selector.note_failure(strategy)

    assert cold(selector).strategy in STRATEGIES


def test_success_clears_the_failure_streak(selector):
    selector.note_failure("SHOCK")
    selector.note_failure("SHOCK")
    selector.note_success("SHOCK")
    assert selector.consecutive_fails["SHOCK"] == 0


# --- межі ваг ---------------------------------------------------------

def test_reward_is_capped(selector):
    cold(selector)
    selector.reward("SHOCK", 99.0)
    assert selector.wins["cold"]["SHOCK"] == StrategySelector.WIN_CEILING


def test_penalty_has_a_lower_floor_than_ordinary_decay(selector):
    """Покарана стратегія падає нижче звичайної підлоги — але не в нуль."""
    cold(selector)
    for _ in range(50):
        selector.penalise("SHOCK", StrategySelector.FAILED_KICK_PENALTY)

    assert selector.wins["cold"]["SHOCK"] == StrategySelector.PENALTY_FLOOR
    assert StrategySelector.PENALTY_FLOOR < StrategySelector.WIN_FLOOR


def test_decaying_others_never_drops_them_below_the_floor(selector):
    cold(selector)
    for _ in range(50):
        selector.decay_others("SHOCK", StrategySelector.OTHERS_DECAY_ON_RECORD)

    assert selector.wins["cold"]["TRIM"] == StrategySelector.WIN_FLOOR
    assert selector.wins["cold"]["SHOCK"] == 1.0     # сама стратегія не чіпається


def test_escape_decays_both_tables_wins_and_tries(selector):
    for table in (selector.wins, selector.tries):
        for phase in ("cold", "hot"):
            table[phase]["SHOCK"] = 4.0

    selector.reset_after_escape()

    for table in (selector.wins, selector.tries):
        for phase in ("cold", "hot"):
            assert table[phase]["SHOCK"] == pytest.approx(4.0 * StrategySelector.ESCAPE_DECAY)


def test_escape_decay_respects_the_floor(selector):
    selector.reset_after_escape()
    assert selector.wins["cold"]["SHOCK"] == StrategySelector.WIN_FLOOR


# --- формули нагород --------------------------------------------------

def test_record_reward_scales_with_the_relative_gain():
    assert StrategySelector.record_reward(0.0) == 1.0
    assert StrategySelector.record_reward(0.01) == pytest.approx(2.0)


def test_diversity_reward_needs_both_distance_and_temperature():
    assert StrategySelector.diversity_reward(0.0, 1.0) == 0.0
    assert StrategySelector.diversity_reward(0.5, 0.0) == 0.0
    assert StrategySelector.diversity_reward(0.5, 1.0) == pytest.approx(0.25)
