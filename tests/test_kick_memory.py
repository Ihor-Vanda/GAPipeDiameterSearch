"""Коротка пам'ять кіків.

Найважливіше тут — що `clear()` справді забуває **всі чотири** пам'яті.
До етапу 8 вони очищались чотирма окремими рядками в гілці `ESCAPE`, і
забути одну з них означало тихий дефект: кік продовжував вважати трубу
безнадійною вже в іншому басейні.
"""

import pytest

from pipeopt.solver.kick_memory import KickMemory


@pytest.fixture
def memory():
    return KickMemory()


def fill(memory):
    memory.bottleneck_failed[1] = 10
    memory.bottleneck_boosted[2] = 11
    memory.loop_balance_failed[3] = 12
    memory.zero_sum_tabu[4] = 13


# --- очищення ---------------------------------------------------------

def test_clear_forgets_every_memory(memory):
    fill(memory)
    memory.clear()

    assert memory.bottleneck_failed == {}
    assert memory.bottleneck_boosted == {}
    assert memory.loop_balance_failed == {}
    assert memory.zero_sum_tabu == {}


def test_clear_covers_all_declared_memories(memory):
    """Якщо додасться п'ята пам'ять, а `clear` про неї не знатиме — впаде тут."""
    fill(memory)
    memory.clear()
    for name in KickMemory.__slots__:
        assert getattr(memory, name) == {}, name


# --- яку пам'ять читає яка стратегія ----------------------------------

def test_bottleneck_reads_its_own_failures(memory):
    assert memory.failed_pipes_for("BOTTLENECK") is memory.bottleneck_failed


def test_loop_balance_reads_its_own_failures(memory):
    assert memory.failed_pipes_for("LOOP_BALANCE") is memory.loop_balance_failed


@pytest.mark.parametrize("strategy", ["SHOCK", "TRIM", "RUIN_RECREATE", "BASIN_ESCAPE"])
def test_other_strategies_get_the_loop_balance_memory(strategy):
    """Так було в тернарному виразі: усе, що не BOTTLENECK, читає LOOP_BALANCE."""
    memory = KickMemory()
    assert memory.failed_pipes_for(strategy) is memory.loop_balance_failed


# --- запис ------------------------------------------------------------

def test_failure_is_recorded_with_the_round_number(memory):
    memory.note_failure("BOTTLENECK", pipe_idx=7, round_idx=42)
    assert memory.bottleneck_failed == {7: 42}


def test_failure_of_an_untracked_strategy_is_not_recorded(memory):
    """Запис для стратегій, які цю пам'ять не читають, був би марним."""
    memory.note_failure("SHOCK", pipe_idx=7, round_idx=42)
    assert memory.loop_balance_failed == {}
    assert memory.bottleneck_failed == {}


def test_only_bottleneck_records_boosted_pipes(memory):
    memory.note_boost("BOTTLENECK", pipe_idx=5, round_idx=3)
    memory.note_boost("SHOCK", pipe_idx=6, round_idx=3)
    assert memory.bottleneck_boosted == {5: 3}


def test_zero_sum_locks_one_pipe_of_the_locked_set(memory):
    memory.note_zero_sum("ZERO_SUM", locked={9}, round_idx=4)
    assert memory.zero_sum_tabu == {9: 4}


def test_zero_sum_with_nothing_locked_records_nothing(memory):
    memory.note_zero_sum("ZERO_SUM", locked=set(), round_idx=4)
    assert memory.zero_sum_tabu == {}


def test_other_strategies_never_touch_the_zero_sum_tabu(memory):
    memory.note_zero_sum("TRIM", locked={9}, round_idx=4)
    assert memory.zero_sum_tabu == {}


# --- аргументи кіка ---------------------------------------------------

def test_kick_args_carry_the_three_memories_a_kick_reads(memory):
    args = memory.as_kick_args("BOTTLENECK")
    assert set(args) == {"failed_pipes", "boosted_pipes", "zero_sum_tabu"}
    assert args["failed_pipes"] is memory.bottleneck_failed
    assert args["boosted_pipes"] is memory.bottleneck_boosted
    assert args["zero_sum_tabu"] is memory.zero_sum_tabu


def test_kick_args_switch_the_failure_memory_by_strategy(memory):
    assert (memory.as_kick_args("LOOP_BALANCE")["failed_pipes"]
            is memory.loop_balance_failed)
    assert (memory.as_kick_args("BOTTLENECK")["failed_pipes"]
            is memory.bottleneck_failed)


def test_kick_args_pass_the_live_dictionaries_not_copies(memory):
    """Кіки дописують у пам'ять на місці, тому копії зламали б поведінку."""
    args = memory.as_kick_args("BOTTLENECK")
    args["failed_pipes"][1] = 99
    assert memory.bottleneck_failed == {1: 99}
