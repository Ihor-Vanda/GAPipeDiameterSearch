"""Самоадаптація розміру кроку кіків (JADE).

Найважливіше тут — що дві політики звуження **не однакові**: вибух
вартості звужує крок без підлоги, звичайний відкид — з підлогою. У
початковому коді це були два різні фрагменти в різних гілках `_apply_kick`,
і різницю між ними було видно лише при уважному читанні.
"""

import pytest

from analytical_solver.adaptive_rates import AdaptiveRates


def test_starts_at_the_documented_defaults():
    rates = AdaptiveRates()
    assert rates.as_kick_args() == {
        "mu_ruin_pct": 0.05, "mu_perturb_pct": 0.10,
        "mu_spatial_pct": 0.20, "mu_escape_pct": 0.20,
    }


@pytest.mark.parametrize("strategy", ["RUIN_RECREATE", "SMART_PERTURB", "SPATIAL_PERTURB"])
def test_size_driven_strategies_are_tracked(strategy):
    assert AdaptiveRates().tracks(strategy) is True


@pytest.mark.parametrize("strategy", ["SHOCK", "BOTTLENECK", "BASIN_ESCAPE", "TRIM"])
def test_other_strategies_have_no_step_size(strategy):
    assert AdaptiveRates().tracks(strategy) is False


# --- навчання ---------------------------------------------------------

def test_learning_moves_the_step_towards_what_was_used():
    rates = AdaptiveRates()
    rates.learn("SMART_PERTURB", used_pct=0.50, alpha=0.20)

    # старт 0.10, зсув на 20 % до 0.50
    assert rates.values["perturb"] == pytest.approx(0.18)


def test_learning_leaves_other_strategies_alone():
    rates = AdaptiveRates()
    rates.learn("SMART_PERTURB", used_pct=0.90, alpha=0.50)

    assert rates.values["ruin"] == 0.05
    assert rates.values["spatial"] == 0.20
    assert rates.escape == 0.20


def test_untracked_strategy_learns_nothing():
    rates = AdaptiveRates()
    rates.learn("SHOCK", used_pct=0.90, alpha=0.50)
    assert rates.snapshot() == AdaptiveRates().snapshot()


def test_zero_alpha_changes_nothing():
    """`pool_alpha` віддає 0.0 поза вікном близькості — це мусить бути no-op."""
    rates = AdaptiveRates()
    rates.learn("RUIN_RECREATE", used_pct=0.90, alpha=0.0)
    assert rates.values["ruin"] == 0.05


def test_escape_learns_at_a_fixed_rate():
    rates = AdaptiveRates()
    rates.learn_escape(0.40)
    # старт 0.20, зсув на 10 % до 0.40
    assert rates.escape == pytest.approx(0.22)


# --- альфи ------------------------------------------------------------

def test_a_small_step_is_trusted_more_than_a_large_one():
    """Великий крок міг дати виграш випадково, тому вчить слабше."""
    assert AdaptiveRates.record_alpha(0.05) > AdaptiveRates.record_alpha(0.80)


def test_record_alpha_never_exceeds_the_base_rate():
    for used in (0.0, 0.01, 0.5, 1.0):
        assert 0.0 < AdaptiveRates.record_alpha(used) <= AdaptiveRates.RECORD_ALPHA


def test_pool_alpha_is_strongest_at_the_record_level():
    assert AdaptiveRates.pool_alpha(0.0) == pytest.approx(AdaptiveRates.POOL_ALPHA)


def test_pool_alpha_fades_across_the_window_and_stops_at_its_edge():
    assert AdaptiveRates.pool_alpha(0.05) == pytest.approx(AdaptiveRates.POOL_ALPHA * 0.5)
    assert AdaptiveRates.pool_alpha(0.10) == 0.0
    assert AdaptiveRates.pool_alpha(0.50) == 0.0


# --- звуження: дві різні політики -------------------------------------

def test_explosion_shrinks_without_a_floor():
    """Вибух вартості означає, що крок явно завеликий — межа не тримає."""
    rates = AdaptiveRates()
    for _ in range(40):
        rates.shrink_after_explosion("RUIN_RECREATE")

    floor = AdaptiveRates.STEP_SIZES["RUIN_RECREATE"][2]
    assert rates.values["ruin"] < floor


def test_reject_shrinks_only_down_to_the_floor():
    rates = AdaptiveRates()
    for _ in range(200):
        rates.shrink_after_reject("RUIN_RECREATE")

    assert rates.values["ruin"] == AdaptiveRates.STEP_SIZES["RUIN_RECREATE"][2]


@pytest.mark.parametrize("strategy,floor", [
    ("RUIN_RECREATE", 0.02), ("SMART_PERTURB", 0.03), ("SPATIAL_PERTURB", 0.05),
])
def test_each_strategy_keeps_its_own_floor(strategy, floor):
    rates = AdaptiveRates()
    for _ in range(200):
        rates.shrink_after_reject(strategy)

    key = AdaptiveRates.STEP_SIZES[strategy][0]
    assert rates.values[key] == floor


def test_one_shrink_applies_the_documented_factors():
    rates = AdaptiveRates()
    rates.shrink_after_explosion("SPATIAL_PERTURB")
    assert rates.values["spatial"] == pytest.approx(0.20 * 0.90)

    fresh = AdaptiveRates()
    fresh.shrink_after_reject("SPATIAL_PERTURB")
    assert fresh.values["spatial"] == pytest.approx(0.20 * 0.95)


def test_shrinking_an_untracked_strategy_is_a_no_op():
    rates = AdaptiveRates()
    rates.shrink_after_explosion("SHOCK")
    rates.shrink_after_reject("BASIN_ESCAPE")
    assert rates.snapshot() == AdaptiveRates().snapshot()
