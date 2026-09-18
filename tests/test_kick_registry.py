"""Реєстр стратегій: доказ, що заміна ланцюга `elif` нічого не змінила.

Тести нижче фіксують пули точно в тому вигляді, в якому вони були
захардкоджені в `_apply_kick` до етапу 3. Порядок елементів має значення:
вибір стратегії йде через `random.choice(...)` і `max(..., key=...)` по
цих списках, тому інший порядок означав би іншу траєкторію пошуку.
"""

import pytest

from pipeopt.solver.kick_registry import (
    COLD_POOL,
    FALLBACK_KICK,
    HOT_THRESHOLD,
    KICK_REGISTRY,
    TRACKED_STRATEGIES,
    WARM_POOL,
    WARM_THRESHOLD,
    KickResult,
    pool_for,
)
from pipeopt.solver.kicks import KickStrategies

# Пули дослівно з коду до етапу 3 (orchestrator._apply_kick, рядки 660-669).
LEGACY_COLD = ["SHOCK", "BOTTLENECK", "LOOP_BALANCE", "ZERO_SUM", "TRIM"]
LEGACY_WARM = ["SHOCK", "BOTTLENECK", "LOOP_BALANCE", "ZERO_SUM", "TRIM",
               "SPATIAL_PERTURB", "SMART_PERTURB", "RUIN_RECREATE", "TOPO_INV"]
LEGACY_HOT_WITH_ARCHIVE = ["BASIN_ESCAPE"]
LEGACY_HOT_NO_ARCHIVE = ["RUIN_RECREATE", "SPATIAL_PERTURB", "TOPO_INV"]


@pytest.mark.parametrize(
    "temperature,archive_size,expected",
    [
        (0.00, 0, LEGACY_COLD),
        (0.39, 5, LEGACY_COLD),
        (WARM_THRESHOLD, 0, LEGACY_WARM),
        (0.89, 5, LEGACY_WARM),
        (HOT_THRESHOLD, 2, LEGACY_HOT_WITH_ARCHIVE),
        (1.00, 6, LEGACY_HOT_WITH_ARCHIVE),
        (HOT_THRESHOLD, 1, LEGACY_HOT_NO_ARCHIVE),
        (1.00, 0, LEGACY_HOT_NO_ARCHIVE),
    ],
)
def test_pool_matches_legacy_hardcoded_lists(temperature, archive_size, expected):
    assert pool_for(temperature, archive_size, is_tree=False) == expected


def test_tree_removes_only_loop_balance():
    looped = pool_for(0.0, 0, is_tree=False)
    tree = pool_for(0.0, 0, is_tree=True)

    assert "LOOP_BALANCE" in looped
    assert "LOOP_BALANCE" not in tree
    assert tree == [s for s in looped if s != "LOOP_BALANCE"]


def test_only_loop_balance_needs_cycles():
    needs = {n for n, spec in KICK_REGISTRY.items() if spec.needs_cycles}
    assert needs == {"LOOP_BALANCE"}


def test_every_spec_method_exists():
    for name, spec in KICK_REGISTRY.items():
        assert hasattr(KickStrategies, spec.method), f"{name}: немає методу {spec.method}"


def test_fallback_is_registered():
    assert FALLBACK_KICK in KICK_REGISTRY


def test_every_pooled_strategy_is_registered_and_tracked():
    pooled = set(COLD_POOL) | set(WARM_POOL) | {"BASIN_ESCAPE"}
    assert pooled <= set(KICK_REGISTRY)
    assert pooled <= set(TRACKED_STRATEGIES)


def test_tracked_has_no_undispatched_strategies():
    """До етапу 3 у статистику UCB1 входила стратегія "SUBMARINE", якої
    немає в диспетчері — їй марно виділялися ваги й спроби."""
    assert set(TRACKED_STRATEGIES) == set(KICK_REGISTRY)
    assert "SUBMARINE" not in TRACKED_STRATEGIES


def test_adaptive_rates_cover_exactly_the_jade_strategies():
    rates = {n: spec.adaptive_rate for n, spec in KICK_REGISTRY.items() if spec.adaptive_rate}
    assert rates == {
        "SMART_PERTURB": "perturb",
        "SPATIAL_PERTURB": "spatial",
        "RUIN_RECREATE": "ruin",
        "BASIN_ESCAPE": "escape",
    }


# --- контракт результату -------------------------------------------------

def test_failure_is_detected():
    assert KickResult.failure().failed
    assert KickResult.failure("no targets").message == "no targets"


def test_success_requires_both_solution_and_locked():
    assert not KickResult([1, 2, 3], set()).failed
    assert KickResult([1, 2, 3], None).failed
    assert KickResult(None, set()).failed


def test_empty_locked_set_is_still_success():
    """Відповідає старій перевірці `forced_sol is None or locked is None`:
    порожня множина заблокованих труб не означала невдачі."""
    assert not KickResult([1, 2, 3], set()).failed


def test_optional_fields_default_to_none():
    r = KickResult([1], set())
    assert r.failed_pipe is None
    assert r.path_signature is None
    assert r.used_pct is None


def test_failed_pipe_zero_is_a_real_pipe():
    """Старий сентинел був -1, тому труба 0 могла б помилково читатись як
    «немає труби», якби перевірка була на falsy."""
    assert KickResult.failure(failed_pipe=0).failed_pipe == 0
    assert KickResult.failure(failed_pipe=0).failed_pipe is not None
