"""`_emergency_respawn` — аварійний вихід із тупика променевого пошуку.

Метод не мав жодного тесту, і саме тому в ньому півроку жив дефект: після
етапу 3 кіки повертають `KickResult`, а тут лишилась перевірка
`isinstance(res, tuple)`, яка для dataclass завжди хибна. Обидва рятувальні
збурення виконувались, витрачали симуляції — і їхній результат мовчки
викидався, а пул наповнював найгрубіший запасний варіант («підняти одну
випадкову трубу на два діаметри»).

Тести нижче фіксують контракт у термінах поведінки, а не реалізації:
рішення, яке дав кік, **мусить потрапити в пул**. Тому вони так само
впадуть, якщо контракт `KickResult` зміниться знову й цей виклик знову
забудуть.
"""

import random

import pytest

from pipeopt.solver.kick_registry import KickArgs, KickResult
from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch
from pipeopt.solver.island import IslandWorker


@pytest.fixture
def worker(ctx):
    random.seed(11)
    local_search = LocalSearch(ctx)
    w = IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                     0, 1, 100_000, 5, [])
    w.run_best_sol = [2] * ctx.num_pipes
    w.run_best_cost = ctx.get_cached_stats(w.run_best_sol).cost
    w.base_dyn_bonus = w.run_best_cost * 0.001
    w.pool.active_pool = [(0.0, w.run_best_cost, list(w.run_best_sol))]
    return w


def pool_solutions(worker):
    return [tuple(sol) for _, _, sol in worker.pool.active_pool]


class FixedKick:
    """Кік, який повертає наперед задані результати по черзі."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def smart_perturbation_kick(self, solution, args):
        self.calls.append(args.T)
        return self.results.pop(0) if self.results else KickResult.failure()


def test_the_first_rescue_kick_is_adopted_into_the_pool(worker, ctx):
    rescue = [1] + [2] * (ctx.num_pipes - 1)
    worker.kicker = FixedKick(KickResult(rescue, set()))

    worker._emergency_respawn()

    assert tuple(rescue) in pool_solutions(worker), (
        "рішення, яке повернув перший кік, мусить потрапити в пул")


def test_the_first_rescue_kick_stops_the_search_for_a_second_one(worker, ctx):
    rescue = [1] + [2] * (ctx.num_pipes - 1)
    kicker = FixedKick(KickResult(rescue, set()))
    worker.kicker = kicker

    worker._emergency_respawn()

    assert kicker.calls == [1.0], (
        "другий, м'якший кік потрібен лише тоді, коли перший нічого не дав")


def test_a_failed_first_kick_falls_back_to_the_softer_one(worker, ctx):
    softer = [0] + [2] * (ctx.num_pipes - 1)
    kicker = FixedKick(KickResult.failure(), KickResult(softer, set()))
    worker.kicker = kicker

    worker._emergency_respawn()

    assert kicker.calls == [1.0, 0.6]
    assert tuple(softer) in pool_solutions(worker)


def test_the_crude_fallback_runs_only_when_both_kicks_fail(worker, ctx):
    worker.kicker = FixedKick(KickResult.failure(), KickResult.failure())
    before = list(worker.run_best_sol)

    worker._emergency_respawn()

    adopted = pool_solutions(worker)
    assert adopted, "пул не може лишитись порожнім після аварійного відновлення"
    # Запасний варіант — це рекорд із однією трубою, піднятою на два кроки.
    differences = [i for i, (a, b) in enumerate(zip(adopted[0], before)) if a != b]
    assert len(differences) <= 1, (
        "запасний варіант міняє щонайбільше одну трубу; якщо змінилось більше — "
        "у пул потрапило щось інше")


def test_a_rescue_that_breaks_the_pressure_limit_is_not_adopted(worker, ctx):
    """Кік може дати недопустиме рішення — його в пул пускати не можна."""
    infeasible = [0] * ctx.num_pipes
    assert not ctx.get_cached_stats(infeasible).is_feasible(ctx.h_min), (
        "передумова тесту: всі мінімальні діаметри не проходять за тиском")
    worker.kicker = FixedKick(KickResult(infeasible, set()),
                              KickResult.failure())

    worker._emergency_respawn()

    assert tuple(infeasible) not in pool_solutions(worker)


def test_the_pool_and_the_exact_tabu_are_flushed(worker):
    worker.pool.tabu_fingerprints[("fake",)] = 1.0
    worker.kicker = FixedKick(KickResult.failure(), KickResult.failure())

    worker._emergency_respawn()

    assert ("fake",) not in worker.pool.tabu_fingerprints, (
        "аварійне відновлення існує саме для того, щоб зняти табу, "
        "яке заблокувало промінь")


def test_a_kick_result_is_not_a_tuple(ctx):
    """Регресія на конкретний дефект: саме ця перевірка стояла у воркері.

    Тест навмисне сформульований про `KickResult`, а не про воркер: він
    фіксує причину, а не наслідок, тому лишається читабельним і тоді,
    коли `_emergency_respawn` буде переписаний.
    """
    local_search = LocalSearch(ctx)
    result = KickStrategies(ctx, local_search).smart_perturbation_kick(
        [2] * ctx.num_pipes, KickArgs(T=1.0))

    assert not isinstance(result, tuple), (
        "кіки повертають KickResult; будь-яка перевірка на кортеж у місці "
        "виклику буде завжди хибною і тихо викине результат")
    assert hasattr(result, "failed") and hasattr(result, "solution")
