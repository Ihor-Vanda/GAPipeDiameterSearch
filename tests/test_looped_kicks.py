"""Три кіки, які ланцюгова тестова мережа приховувала повністю.

Покриття `kicks_structural.py` було **53 %**, `kicks_path.py` — 73 %, і
причина була не в тому, що тестів мало. Тести були: вони викликали кік і
перевіряли, що він не впав. Але `FakeSimulator` — послідовний ланцюг
`R -> J1 -> ... -> Jn`, а всі три стратегії нижче спираються на те, чого
в ланцюзі не буває:

| стратегія | вихід на ланцюзі | рядків не виконувалось |
|---|---|---|
| `LOOP_BALANCE` | `nx.cycle_basis` порожній -> «No cycles found» | 55 |
| `TRIM` | домінантний шлях = вся мережа -> периферія порожня | 41 |
| `TOPO_INV` | єдиний шлях = домінантний -> «All paths tabu» | 35 |

Тобто **сто тридцять рядків трьох стратегій** не виконувались жодним
тестом, а тести при цьому були зелені — вони перевіряли поведінку
«кік чесно повідомив, що йому нічого робити». Це найдорожчий різновид
хибного покриття: ані `pytest`, ані `coverage` не скажуть, що тест
перевіряє не те, заради чого написаний.

`LoopedSimulator` (див. `conftest.py`) дає два цикли й тупикову вітку.
Нижче перевіряється не «не впало», а зміст кожної стратегії: **що саме
вона має право змінити** і **коли мусить відмовитись**.
"""

import random

import networkx as nx
import pytest

from pipeopt.solver.kick_registry import KickArgs
from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch

#: Допустиме рішення, у якому магістраль (P1-P3) на максимумі, а труби
#: циклів — низько. Це єдина форма, у якій усі три стратегії мають що
#: робити одночасно: периферія непорожня (потрібно `indices[i] <= 1`),
#: труби циклів можна і підняти, і опустити.
FEASIBLE = [3, 3, 3, 1, 1, 1, 1, 1, 1]

#: Магістраль: домінантний шлях від джерела до критичного вузла G.
TRUNK = (0, 1, 2, 5, 8)
#: Труби циклів, яких домінантний шлях не торкається.
PERIPHERY = (3, 4, 6, 7)


@pytest.fixture
def kicker(looped_ctx):
    return KickStrategies(looped_ctx, LocalSearch(looped_ctx))


@pytest.fixture
def args():
    """Раунд 100 і порожня пам'ять: жодна труба ще не в tabu."""
    return KickArgs(T=0.5, current_round=100)


@pytest.fixture(autouse=True)
def fixed_seed():
    """Усі три стратегії тягнуть `random.sample` — без зерна тест мерехтить."""
    random.seed(20260915)


def cycles_of(ctx):
    """Труби кожного циклу — те саме, що бачить `loop_balancing_kick`."""
    out = []
    for cycle_nodes in nx.cycle_basis(ctx.base_G_flow):
        ring = cycle_nodes + [cycle_nodes[0]]
        out.append({ctx.edge_to_pipe[(u, v)] for u, v in zip(ring[:-1], ring[1:])
                    if (u, v) in ctx.edge_to_pipe})
    return out


# --- сама мережа: фікстура мусить мати те, заради чого її написано -----

def test_the_looped_network_really_has_cycles(looped_ctx):
    """Якщо ця перевірка впаде, решта файлу перевіряє порожні гілки."""
    assert len(cycles_of(looped_ctx)) == 2


def test_the_dominant_path_does_not_cover_the_whole_network(looped_ctx):
    """Саме це й ховала ланцюгова мережа: у ланцюзі периферії не буває."""
    dominant, _ = looped_ctx.get_dominant_path(
        FEASIBLE, looped_ctx.get_cached_stats(FEASIBLE)[3])

    assert set(dominant) == set(TRUNK)
    assert set(looped_ctx.get_lazy_periphery(FEASIBLE)[1]) == set(PERIPHERY)


def test_the_starting_solution_is_feasible(looped_ctx):
    """Кіки приймають лише допустимі кандидати — старт мусить бути таким."""
    stats = looped_ctx.get_cached_stats(FEASIBLE)

    assert stats[1] >= looped_ctx.h_min


# --- LOOP_BALANCE: перерозподіл усередині циклу ------------------------

def test_loop_balancing_cuts_pipes_inside_a_single_cycle(kicker, looped_ctx, args):
    """Сенс стратегії: звузити частину циклу й змусити потік піти в обхід.

    Тому заблоковані труби мусять належати **одному** циклу: перерізати
    по трубі в двох різних циклах — це вже не «перерозподіл у циклі».
    """
    result = kicker.loop_balancing_kick(FEASIBLE, args)

    assert not result.failed, result.message
    assert result.locked
    assert any(result.locked <= cycle for cycle in cycles_of(looped_ctx)), (
        f"заблоковані труби {sorted(result.locked)} не лежать в одному циклі")


def test_loop_balancing_only_narrows_the_pipes_it_locks(kicker, args):
    """Заблоковані труби — це саме ті, які кік звузив, і лише звузив."""
    result = kicker.loop_balancing_kick(FEASIBLE, args)

    assert not result.failed, result.message
    for pipe in result.locked:
        assert result.solution[pipe] < FEASIBLE[pipe], (
            f"труба {pipe} заблокована, але не звужена")


def test_loop_balancing_returns_a_feasible_solution(kicker, looped_ctx, args):
    """Кандидат приймається лише після `heal_network`, тобто допустимим.

    Це і є ціна стратегії: вона витрачає симуляції на лікування, зате
    ніколи не повертає недопустиме рішення.
    """
    result = kicker.loop_balancing_kick(FEASIBLE, args)

    assert not result.failed, result.message
    assert looped_ctx.get_cached_stats(result.solution)[1] >= looped_ctx.h_min


def test_loop_balancing_reports_the_pipe_it_worked_on(kicker, args):
    """Воркер заносить цю трубу в tabu — без неї кік ходив би по колу."""
    result = kicker.loop_balancing_kick(FEASIBLE, args)

    assert result.failed_pipe is not None
    assert 0 <= result.failed_pipe < len(FEASIBLE)


def test_loop_balancing_says_what_it_did(kicker, args):
    result = kicker.loop_balancing_kick(FEASIBLE, args)

    assert "FLOW STEER" in result.message
    assert "Healed" in result.message, "звіт мусить показати ціну лікування"


def test_loop_balancing_skips_pipes_it_recently_failed_on(kicker, looped_ctx):
    """Пам'ять невдач: щойно провалена труба не береться повторно.

    Термін tenure — `min(80, max(3, num_pipes // 5))`; на дев'яти трубах
    це 3 раунди. Позначаємо всі труби циклів як щойно провалені й
    перевіряємо, що кік відмовляється, а не бере їх знову.
    """
    just_failed = {pipe: 100 for cycle in cycles_of(looped_ctx) for pipe in cycle}

    result = kicker.loop_balancing_kick(
        FEASIBLE, KickArgs(T=0.5, current_round=100, failed_pipes=just_failed))

    assert result.failed
    assert "no valid bypass" in result.message.lower()


def test_loop_balancing_refuses_pipes_that_are_already_narrow(kicker, args):
    """Труби з індексом < 2 не ріжуться: звужувати мінімум нікуди.

    Перевірка живе в коді як `if any(indices[p] < 2 ...): continue`, і
    без циклів у мережі вона ніколи не виконувалась.
    """
    all_narrow = [1] * 9

    result = kicker.loop_balancing_kick(all_narrow, args)

    assert result.failed


def test_loop_balancing_gives_up_when_the_network_has_no_cycles(sim, args):
    """Зворотний бік: на ланцюзі стратегія мусить чесно відмовитись.

    Це та сама відповідь, яку раніше отримували **всі** тести цієї
    стратегії, — тепер вона перевіряється навмисно й лише тут.
    """
    from pipeopt.solver.context import SolverContext

    chain_ctx = SolverContext(sim, sim.config.diameters_m, v_opt=1.0)
    chain_kicker = KickStrategies(chain_ctx, LocalSearch(chain_ctx))

    result = chain_kicker.loop_balancing_kick([2] * sim.n_variables, args)

    assert result.failed
    assert result.message == "No cycles found"


def test_loop_balancing_computes_the_cycle_basis_only_once(kicker, args):
    """Цикли кешуються на весь прогін: топологія не змінюється.

    Перевірка змістова, а не про швидкодію: якби кеш перечитувався,
    підміна тут не мала б жодного ефекту, і кожен кік платив би
    `cycle_basis` на мережі з сотень труб.
    """
    kicker.loop_balancing_kick(FEASIBLE, args)
    kicker._cached_cycles = []

    assert kicker.loop_balancing_kick(FEASIBLE, args).message == "No cycles found"


# --- TRIM: підрізання периферії ----------------------------------------

def test_trim_only_touches_pipes_off_the_dominant_path(kicker, args):
    """Уся суть стратегії: магістраль недоторкана, економія — з периферії.

    Зачепити домінантний шлях означає збити тиск у критичному вузлі —
    те, чого TRIM саме й уникає, на відміну від RUIN і SHOCK.
    """
    result = kicker.peripheral_trim_kick(FEASIBLE, args)

    assert not result.failed, result.message
    assert result.locked
    assert result.locked <= set(PERIPHERY), (
        f"TRIM зачепив магістраль: {sorted(result.locked - set(PERIPHERY))}")


def test_trim_shrinks_each_pipe_by_exactly_one_step(kicker, args):
    """`test_sol[p] -= 1`: TRIM підрізає обережно, на один діаметр."""
    result = kicker.peripheral_trim_kick(FEASIBLE, args)

    assert not result.failed, result.message
    for pipe in result.locked:
        assert result.solution[pipe] == FEASIBLE[pipe] - 1


def test_trim_returns_a_feasible_and_cheaper_solution(kicker, looped_ctx, args):
    result = kicker.peripheral_trim_kick(FEASIBLE, args)

    assert not result.failed, result.message
    cost, pressure, _, _ = looped_ctx.get_cached_stats(result.solution)
    assert pressure >= looped_ctx.h_min
    assert cost < looped_ctx.get_cached_stats(FEASIBLE)[0], (
        "підрізання периферії мусить давати економію — інакше воно безцільне")


def test_trim_says_how_many_pipes_it_shrank(kicker, args):
    result = kicker.peripheral_trim_kick(FEASIBLE, args)

    assert result.message.startswith("TRIM: Shrunk")
    assert f"Shrunk {len(result.locked)} " in result.message


def test_trim_gives_up_when_there_is_no_periphery(kicker, args):
    """Уся мережа широка -> `indices[i] <= threshold` не виконується ніде.

    `threshold_idx = int(max_d_idx * 0.6)` — на таблиці з чотирьох
    діаметрів це індекс 1, тому рішення з усіх «2» не має периферії.
    """
    result = kicker.peripheral_trim_kick([2] * 9, args)

    assert result.failed
    assert result.message == "No targets"


def test_trim_refuses_a_periphery_that_is_already_at_the_minimum(kicker, args):
    """Труби на нульовому індексі підрізати нікуди: `valid_combo` лишається False."""
    trunk_high_periphery_zero = [3, 3, 3, 0, 0, 0, 0, 0, 3]

    result = kicker.peripheral_trim_kick(trunk_high_periphery_zero, args)

    assert result.failed


# --- TOPO_INV: підсилення альтернативного шляху ------------------------

def test_topological_inversion_picks_a_path_that_is_not_the_dominant_one(
        kicker, looped_ctx, args):
    """Стратегія існує лише там, де до критичного вузла є другий шлях.

    На ланцюзі другого шляху немає за побудовою, тому до появи цієї
    мережі стратегія не перевірялась жодного разу.

    Перевіряється саме **обраний шлях**, а не заблоковані труби: шляхи
    частково збігаються (у звіті кіка це `Overlap`), тому окрема
    піднята труба цілком може лежати й на домінантному шляху.
    """
    dominant, _ = looped_ctx.get_dominant_path(
        FEASIBLE, looped_ctx.get_cached_stats(FEASIBLE)[3])

    result = kicker.topological_inversion_kick(FEASIBLE, args)

    assert not result.failed, result.message
    assert set(result.path_signature) != set(dominant), (
        "обрано домінантний шлях — це не інверсія")
    assert result.locked <= set(result.path_signature), (
        f"підняті труби {sorted(result.locked)} не лежать на обраному шляху "
        f"{sorted(result.path_signature)}")


def test_topological_inversion_never_narrows_a_pipe(kicker, args):
    """TOPO_INV лише піднімає діаметри: він шукає пропускну здатність."""
    result = kicker.topological_inversion_kick(FEASIBLE, args)

    assert not result.failed, result.message
    assert all(after >= before
               for after, before in zip(result.solution, FEASIBLE))
    assert result.solution != FEASIBLE, "кік не змінив нічого"


def test_topological_inversion_levels_pipes_up_to_the_trunk_capacity(kicker, args):
    """Мета — не максимум, а «як магістраль»: середнє по домінантному шляху.

    У `FEASIBLE` магістраль на індексі 3, труби циклів на 1; середнє по
    домінантному шляху (3,3,3,1,1) дає цільовий індекс 2.
    """
    result = kicker.topological_inversion_kick(FEASIBLE, args)

    assert not result.failed, result.message
    for pipe in result.locked:
        assert result.solution[pipe] == 2, (
            "труба альтернативного шляху виведена не на рівень магістралі")


def test_topological_inversion_reports_the_path_it_used(kicker, args):
    """Підпис шляху — те, чим воркер наповнює `tabu_set` (§10.3)."""
    result = kicker.topological_inversion_kick(FEASIBLE, args)

    assert result.path_signature is not None
    assert result.path_signature == tuple(sorted(result.path_signature)), (
        "підпис мусить бути відсортованим, інакше два однакові шляхи "
        "дадуть різні підписи й tabu не спрацює")


def test_each_new_call_looks_for_a_path_it_has_not_tried(kicker, args):
    """Без tabu кік повертав би той самий обхід до кінця прогону.

    Перевірка йде до виснаження: щоразу підпис попереднього шляху
    додається в tabu, і кік мусить або знайти **інший** шлях, або
    відмовитись. Обидва варіанти правильні; неправильний — повторити
    підпис, який уже в tabu.
    """
    tried = set()
    for _ in range(10):
        result = kicker.topological_inversion_kick(
            FEASIBLE, KickArgs(T=0.5, current_round=100, tabu_set=set(tried)))
        if result.failed:
            assert result.message in ("All paths tabu",
                                      "TOPO-INV: alt path already at trunk capacity")
            break
        assert result.path_signature not in tried, (
            f"шлях {result.path_signature} уже в tabu, а кік повернув його знову")
        tried.add(result.path_signature)
    else:
        pytest.fail("кік знайшов 10 різних шляхів у мережі з двома циклами")

    assert tried, "жоден виклик не дав шляху — перевіряти нічого"


def test_topological_inversion_gives_up_on_a_chain(sim, args):
    """На ланцюзі єдиний шлях і є домінантним, альтернатив немає."""
    from pipeopt.solver.context import SolverContext

    chain_ctx = SolverContext(sim, sim.config.diameters_m, v_opt=1.0)
    chain_kicker = KickStrategies(chain_ctx, LocalSearch(chain_ctx))

    result = chain_kicker.topological_inversion_kick([2] * sim.n_variables, args)

    assert result.failed
    assert result.message == "All paths tabu"


def test_a_hotter_temperature_touches_at_least_as_many_pipes(kicker):
    """`aggressiveness = 0.10 + 0.50*T` — єдине, чим T керує в цій стратегії.

    Замір, а не одна пара: `random.sample` робить окремий виклик шумним,
    тому порівнюються середні по десяти зернах.
    """
    def average_touched(temperature):
        total = 0
        for seed in range(10):
            random.seed(seed)
            result = kicker.topological_inversion_kick(
                FEASIBLE, KickArgs(T=temperature, current_round=100))
            total += 0 if result.failed else len(result.locked)
        return total / 10

    assert average_touched(1.0) >= average_touched(0.0)


def test_a_kick_that_cannot_raise_anything_reports_failure(kicker, args):
    """Дефект, знайдений цією мережею: «успіх», який нічого не змінив.

    Труби ширші за магістраль не піднімаються жодною гілкою циклу
    присвоєння, і якщо вибірка складається лише з них, кік повертав
    вхідне рішення як успіх. Наслідки — нагорода UCB1 за нічого й
    витрачений цикл `_polish_kicked`.

    Замір на Ханої (120 тис., зерна 42/1/7): 148-153 викликів TOPO_INV
    за прогін, з них 0-2 порожніх. Разом із ним знайдено значно
    масовіший випадок того самого класу — R&R, ~60 % порожніх успіхів;
    обидва описані в `CHANGELOG.md`.
    """
    trunk_low_periphery_high = [1, 1, 1, 3, 3, 3, 3, 3, 3]

    result = kicker.topological_inversion_kick(trunk_low_periphery_high, args)

    assert result.failed, (
        f"кік повідомив успіх: {result.message!r}, "
        f"рішення {result.solution}")
