"""Порожній кік мусить повідомляти невдачу, а не «успіх без змін».

Із 36 місць, де `kicks.py` повертає результат, 31 користувалось
`KickResult.failure()`, а п'ять — `KickResult(indices, set())`, тобто
успіхом із рішенням, рівним вхідному. Для `_apply_kick` це різні речі:

* успіх → `selector.note_success`, повний цикл `gradient_squeeze` і
  класифікації — на рішенні, яке ніхто не міняв;
* невдача → `selector.note_failure`, запис у пам'ять кіків, штраф UCB1.

Тобто дві найчастіші стратегії (SHOCK — стратегія за замовчуванням,
BOTTLENECK — у всіх трьох температурних пулах) мали окрему, тихішу й
дорожчу семантику невдачі, і UCB1 нагороджував їх за нічого.
"""

import pathlib
import re

import pytest

from pipeopt.solver.kick_registry import KICK_REGISTRY, KickArgs
from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch


@pytest.fixture
def kicker(ctx):
    return KickStrategies(ctx, LocalSearch(ctx))


def test_shock_reports_failure_when_every_pipe_is_already_at_the_maximum(kicker, ctx):
    at_the_top = [ctx.max_d_idx] * ctx.num_pipes

    result = kicker.forcing_hand_kick(at_the_top, KickArgs(T=0.5))

    assert result.failed, (
        "піднімати нема що — це невдача кіка, а не успіх без змін")
    assert result.message, "невдача мусить пояснювати причину в лог"


def test_bottleneck_reports_failure_when_every_pipe_is_already_at_the_maximum(kicker, ctx):
    at_the_top = [ctx.max_d_idx] * ctx.num_pipes

    result = kicker.upstream_bottleneck_kick(at_the_top, KickArgs(T=0.5))

    assert result.failed


def test_a_successful_kick_actually_changes_the_solution(kicker, ctx):
    room_to_grow = [1] * ctx.num_pipes

    result = kicker.forcing_hand_kick(room_to_grow, KickArgs(T=0.5))

    if not result.failed:
        assert list(result.solution) != room_to_grow, (
            "успіх означає зміну: незмінене рішення мусить бути failure()")
        assert result.locked, "успішний кік фіксує труби, які він змінив"


@pytest.mark.parametrize("method", sorted({s.method for s in KICK_REGISTRY.values()}))
def test_every_strategy_returns_a_kick_result(kicker, ctx, method):
    """Контракт один для всіх десяти стратегій — і це перевіряється."""
    result = getattr(kicker, method)(
        [1] * ctx.num_pipes,
        KickArgs(T=0.5, dyn_bonus=1.0, max_perturb=max(1, ctx.num_pipes // 2)))
    assert not isinstance(result, tuple), "кіки більше не повертають кортежі"
    assert hasattr(result, "failed")
    if not result.failed:
        assert result.solution is not None and result.locked is not None
        assert len(result.solution) == ctx.num_pipes


#: Модулі, у яких живуть стратегії. Після розділення етапу 16 сам
#: `kicks.py` — фасад на 31 рядок, і перевірка, яка читала лише його,
#: стала беззмістовною: вона сканувала файл без жодної стратегії й
#: проходила завжди. Знайдено 15.09 разом із петльовою тестовою мережею.
KICK_MODULES = ("kicks.py", "kicks_path.py", "kicks_structural.py", "kicks_ruin.py")


def test_no_strategy_reports_success_with_an_unchanged_solution():
    """Регресія на джерело дефекту, а не на його наслідок.

    `KickResult(indices, set())` — це синтаксично успіх із вхідним
    рішенням. Поки такий виклик існує в коді, стратегія може тихо
    «вдатись», нічого не зробивши.
    """
    package = pathlib.Path(__file__).resolve().parents[1] / "pipeopt" / "solver"
    offenders = []
    for name in KICK_MODULES:
        source = (package / name).read_text(encoding="utf-8")
        offenders += [f"{name}: {m}" for m in
                      re.findall(r"KickResult\(\s*indices\s*,\s*set\(\)\s*\)",
                                 source)]
    assert not offenders, (
        f"знайдено «порожні успіхи»: {offenders}; для «нема що робити» "
        "існує KickResult.failure()")


def test_the_modules_this_file_scans_are_all_of_them():
    """Без цієї перевірки попередній тест знову тихо перестане перевіряти.

    Саме так він і зламався: стратегії переїхали у три нові модулі,
    перевірка лишилась на `kicks.py`, і жодне падіння про це не
    повідомило — тест став зеленим назавжди.
    """
    package = pathlib.Path(__file__).resolve().parents[1] / "pipeopt" / "solver"
    on_disk = {p.name for p in package.glob("kicks*.py")}

    assert on_disk == set(KICK_MODULES), (
        f"модулі стратегій змінились: на диску {sorted(on_disk)}, "
        f"перевіряються {sorted(KICK_MODULES)}")


@pytest.mark.parametrize("method", sorted({s.method for s in KICK_REGISTRY.values()}))
def test_a_strategy_that_changes_nothing_reports_failure(looped_ctx, method):
    """Контракт етапу 13b, перевірений **прогоном**, а не читанням коду.

    Регексп вище ловить лише дослівний `KickResult(indices, set())`.
    `TOPO_INV` порушував той самий контракт інакше: він збирав `kicked`,
    не змінював у ньому нічого (усі труби альтернативного шляху вже
    ширші за магістраль) і повертав це як успіх. На ланцюговій мережі
    гілка недосяжна, тому дефект прожив до появи `LoopedSimulator`.

    Замір на Ханої, 120 тис. симуляцій, зерно 42: 157 викликів
    `TOPO_INV`, 2 без жодної зміни.
    """
    kicker = KickStrategies(looped_ctx, LocalSearch(looped_ctx))
    args = KickArgs(T=0.5, current_round=100, dyn_bonus=1.0,
                    max_perturb=max(1, looped_ctx.num_pipes // 2))

    for start in ([3] * looped_ctx.num_pipes,
                  [0] * looped_ctx.num_pipes,
                  [3, 3, 3, 1, 1, 1, 1, 1, 1]):
        result = getattr(kicker, method)(start, args)
        if result.failed:
            continue
        assert list(result.solution) != list(start), (
            f"{method} повідомив успіх, не змінивши рішення {start}; "
            "для «нема що робити» існує KickResult.failure()")
