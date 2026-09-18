"""Глибина поліровки кандидата: чому саме стільки проходів.

Внутрішня поліровка в `evaluate_candidate` існує, щоб **розставити
кандидатів по місцях**. Доводити кожного до кінця не потрібно: глибоко
полірує промінь, і лише тих, хто в нього потрапив. Тому тут перевіряється
не число саме по собі, а **співвідношення** двох глибин — якщо колись
зрівняються, поліровка переможців перестане бути окремим кроком.

Замір, який стоїть за числом (§25.7): за однакового бюджету симуляцій
один прохід ніде не гірший, а на великих мережах кращий — бо три проходи
з'їдають кандидатів (на Балермі вдвічі-втричі).
"""

import inspect
import re

import pytest

from pipeopt.solver.island import beam
from pipeopt.solver.local_search import LocalSearch


@pytest.fixture
def ls(ctx):
    return LocalSearch(ctx)


def test_candidate_polish_uses_the_ranking_depth(ls, ctx):
    """Поліровка кандидата має брати глибину з константи, а не з літерала."""
    seen = []
    original = ls.gradient_squeeze

    def spy(indices, **kwargs):
        seen.append(kwargs.get("max_passes"))
        return original(indices, **kwargs)

    ls.gradient_squeeze = spy
    base = [ctx.max_d_idx] * ctx.num_pipes
    # `dyn_bonus=None` тут не годиться: у промені його завжди рахують
    # заздалегідь, і `score` множить його на коефіцієнт.
    score, _, _ = ls.evaluate_candidate(base, [0], "downgrade", 1.0)

    assert seen, "поліровки не було — перевірка нічого не перевірила"
    assert seen == [LocalSearch.RANKING_POLISH_PASSES], (
        "евристика оцінки кандидата мусить полірувати рівно раз і рівно на "
        f"глибину RANKING_POLISH_PASSES, а не {seen}")
    assert score != float("inf"), "кандидат мав бути допустимим"


def test_ranking_polish_is_shallower_than_the_beam_polish():
    """Глибока поліровка живе в промені — і має лишатись глибшою.

    Скан навмисно самоохоронний: якщо виклики поїдуть з
    `_beam_search_and_update`, `depths` спорожніє і тест впаде, а не
    почне мовчки проходити.
    """
    source = inspect.getsource(beam.BeamSearch._beam_search_and_update)
    depths = [int(value) for value in re.findall(r"max_passes=(\d+)", source)]

    assert len(depths) >= 2, (
        "у промені більше немає двох рівнів поліровки переможців — "
        "перевірка зіставляє глибини, яких уже не існує")
    assert min(depths) > LocalSearch.RANKING_POLISH_PASSES, (
        f"поліровка для ранжування ({LocalSearch.RANKING_POLISH_PASSES}) "
        f"мусить бути мілкішою за поліровку переможців ({depths})")


def test_deeper_polish_really_costs_more_simulations(ls, ctx):
    """Ціна глибини — не припущення: її видно на лічильнику симуляцій."""
    start = [ctx.max_d_idx] * ctx.num_pipes

    before = ctx.sim_count
    ls.gradient_squeeze(start, max_passes=LocalSearch.RANKING_POLISH_PASSES,
                        quick_mode=True)
    shallow = ctx.sim_count - before

    before = ctx.sim_count
    ls.gradient_squeeze(start, max_passes=3, quick_mode=True)
    deep = ctx.sim_count - before

    assert shallow > 0, "мілка поліровка не зробила жодної симуляції"
    assert deep > shallow, (
        "три проходи мусять коштувати дорожче за один — інакше замір, "
        "на якому стоїть вибір глибини, не має сенсу")
