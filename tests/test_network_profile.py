"""Профіль мережі: одне джерело всього, що залежить від розміру.

До етапу 8 цих залежностей було вісім у трьох файлах — чотири таблиці
«значення на клас» і три окремих `if num_pipes >= 200`, які випадково
збігалися з межею MEDIUM/LARGE, плюс `LARGE_NETWORK_PIPES` у політиці
прийняття. Тести нижче фіксують, що значення **не змінились** (окрім
свідомо виправленого бюджету) і що три колишні `>= 200` дійсно
еквівалентні `is_large`.
"""

import pytest

from analytical_solver.context import NetworkProfile, beam_width_for, classify_network


# --- класи ------------------------------------------------------------

@pytest.mark.parametrize("num_pipes,expected", [
    (1, "SMALL"), (34, "SMALL"), (49, "SMALL"),
    (50, "MEDIUM"), (150, "MEDIUM"), (199, "MEDIUM"),
    (200, "LARGE"), (454, "LARGE"), (999, "LARGE"),
    (1000, "XLARGE"), (1105, "XLARGE"), (3961, "XLARGE"),
])
def test_class_boundaries_are_unchanged(num_pipes, expected):
    assert NetworkProfile(num_pipes).name == expected
    assert classify_network(num_pipes) == expected


@pytest.mark.parametrize("num_pipes", [1, 34, 199])
def test_small_and_medium_are_not_large(num_pipes):
    assert NetworkProfile(num_pipes).is_large is False


@pytest.mark.parametrize("num_pipes", [200, 454, 1105, 10_000])
def test_large_and_xlarge_are_large(num_pipes):
    assert NetworkProfile(num_pipes).is_large is True


def test_is_large_is_exactly_the_old_two_hundred_pipe_threshold():
    """Три місця в коді мали власний `num_pipes >= 200`; це та сама межа."""
    for num_pipes in range(1, 1200):
        assert NetworkProfile(num_pipes).is_large == (num_pipes >= 200)


# --- значення, які мусили лишитись тими самими -------------------------

@pytest.mark.parametrize("num_pipes,width", [
    (34, 5), (150, 5), (454, 8), (1105, 8),
])
def test_beam_width_unchanged(num_pipes, width):
    assert NetworkProfile(num_pipes).beam_width == width
    assert beam_width_for(NetworkProfile(num_pipes).name) == width


@pytest.mark.parametrize("num_pipes,candidates", [
    (34, 6),        # max(6, 34 // 5)
    (20, 6),        # підлога 6
    (150, 15),      # 150 // 10
    (454, 22),      # 454 // 20
    (1105, 10),     # XLARGE — фіксовано
    (3961, 10),
])
def test_single_candidates_unchanged(num_pipes, candidates):
    assert NetworkProfile(num_pipes).single_candidates == candidates


@pytest.mark.parametrize("num_pipes,passes", [(34, 5), (199, 5), (200, 8), (1105, 8)])
def test_deep_polish_depth_unchanged(num_pipes, passes):
    assert NetworkProfile(num_pipes).deep_polish_passes == passes


@pytest.mark.parametrize("num_pipes,patience", [(34, 10), (199, 10), (454, 6), (1105, 6)])
def test_restart_patience_unchanged(num_pipes, patience):
    assert NetworkProfile(num_pipes).restart_patience == patience


@pytest.mark.parametrize("num_pipes,interval", [
    (34, 1.0), (150, 15.0), (454, 30.0), (1105, 60.0),
])
def test_print_interval_unchanged(num_pipes, interval):
    assert NetworkProfile(num_pipes).print_interval == interval


# --- бюджет: єдина свідома зміна --------------------------------------

def test_hanoi_budget_is_untouched():
    """Еталон Ханої мусить лишитись відтворюваним, тому SMALL не рухаємо."""
    assert NetworkProfile(34).sim_budget == 1_000_000


def test_budget_never_decreases_with_network_size():
    """Було немонотонно: LARGE отримував 1.5 млн проти 3 млн у MEDIUM (§22.4.3)."""
    budgets = [NetworkProfile(n).sim_budget for n in (10, 60, 250, 1500, 5000)]
    assert budgets == sorted(budgets)


def test_large_networks_now_get_at_least_the_medium_budget():
    assert NetworkProfile(454).sim_budget >= NetworkProfile(150).sim_budget
