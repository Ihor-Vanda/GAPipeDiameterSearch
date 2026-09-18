"""Чотири механізми, яких послідовний прогін не вмикає взагалі.

`_process_ipc`, `_check_rescue`, `_check_mini_restart` і `_apply_swap` —
це 110 рядків `island.py`, які виконуються лише коли острови бачать один
одного. Еталон іде з `--sequential`, тобто `shared_progress is None`, і
жоден із них там не виконується жодного разу. Покриття `island.py`
показувало 75 % саме через них.

Небезпека тут не теоретична, і вона однакова в усіх чотирьох: **вони не
падають**. Воркер, який помилково всиновив чуже рішення, не кидає
винятку — він просто перестає шукати своє, і острівна модель тихо
вироджується в один пошук у N копіях. Помітити це в логах прогону
неможливо: усі острови дадуть результат, і він буде правдоподібним.

Тому перевіряються не «не впало», а **умови спрацювання**: коли механізм
має право втрутитись і коли мусить утриматись.
"""

import random

import pytest

from pipeopt.solver.island import IslandWorker
from pipeopt.solver.kicks import KickStrategies
from pipeopt.solver.local_search import LocalSearch

#: Власний рекорд воркера: усі труби на індексі 2.
OWN = 2
#: Дешевше й досі допустиме рішення — те, що приносять сусіди.
PEER = 1


@pytest.fixture
def worker(ctx):
    random.seed(11)
    local_search = LocalSearch(ctx)
    w = IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                     worker_id=0, n_workers=3, max_sims=100_000,
                     beam_width=5, global_archive=[])
    w.run_best_sol = [OWN] * ctx.num_pipes
    w.run_best_cost = ctx.get_cached_stats(w.run_best_sol).cost
    w.base_dyn_bonus = w.run_best_cost * 0.001
    w.pool.active_pool = [(0.0, w.run_best_cost, list(w.run_best_sol))]
    # Імунітет після старту — окремий механізм; тести, яким він заважає,
    # знімають його явно, а два тести нижче перевіряють саме його.
    w.ipc_immunity = 0
    return w


@pytest.fixture
def peer_solution(ctx):
    return [PEER] * ctx.num_pipes


@pytest.fixture
def peer_cost(ctx, peer_solution):
    return ctx.get_cached_stats(peer_solution).cost


def shared(**entries):
    """`shared_progress` — звичайний словник у тестах і Manager().dict() у бою."""
    return dict(entries)


# --- IPC: пасивне підглядання за сусідами ------------------------------

def test_the_immunity_window_blocks_everything_at_the_start(worker, peer_cost,
                                                            peer_solution):
    """Перші 30 раундів воркер не дивиться на сусідів.

    Інакше всі острови стартують з однакового зерна й миттєво сходяться
    до першого ж рекорду — острівна модель не встигає розійтись.
    """
    worker.ipc_immunity = 3
    progress = shared(best_sol_1=(peer_cost, peer_solution))

    worker._process_ipc(progress, (peer_cost, peer_solution))

    assert worker.ipc_immunity == 2, "імунітет мусить спадати по одному за раунд"
    assert worker.global_best_cost == float("inf"), "межа не мала оновитись"
    assert len(worker.pool.active_pool) == 1


def test_a_cheaper_global_bound_is_adopted(worker, peer_cost, peer_solution):
    """Межа — це число, а не рішення: воно лише робить пошук вибагливішим."""
    worker._process_ipc(shared(), (peer_cost, peer_solution))

    assert worker.global_best_cost == peer_cost


def test_a_worse_global_bound_is_ignored(worker):
    worker.global_best_cost = 1.0

    worker._process_ipc(shared(), (999.0, [OWN] * 6))

    assert worker.global_best_cost == 1.0


def test_a_much_better_peer_solution_is_injected_into_the_pool(worker, peer_cost,
                                                               peer_solution):
    """Рішення сусіда потрапляє **в пул**, а не в рекорд.

    Це і є «пасивне» щеплення: воркер отримує матеріал для кіків, але
    його власний рекорд лишається своїм, і острів не здається.
    """
    before = worker.run_best_cost
    progress = shared(best_sol_1=(peer_cost, peer_solution))

    worker._process_ipc(progress, None)

    assert len(worker.pool.active_pool) == 2
    assert worker.pool.active_pool[-1][2] == peer_solution
    assert worker.run_best_cost == before, "рекорд воркера чіпати не можна"


def test_the_same_peer_solution_is_not_injected_twice(worker, peer_cost,
                                                      peer_solution):
    """Без пам'яті `last_injected_peer` пул щороунду набивався б клонами."""
    progress = shared(best_sol_1=(peer_cost, peer_solution))

    worker._process_ipc(progress, None)
    worker._process_ipc(progress, None)

    assert len(worker.pool.active_pool) == 2


def test_a_slightly_better_peer_is_ignored_while_the_search_still_moves(
        worker, peer_solution):
    """Поки воркер не застряг, дрібна перевага сусіда його не стосується.

    Поріг «масово краще» — 2 % рекорду. Нижче нього щеплення дозволене
    лише при застої, інакше острови зрівнялись би на першій же дрібниці.
    """
    barely_better = worker.run_best_cost * 0.99
    worker.stagnation_counter = 0
    progress = shared(best_sol_1=(barely_better, peer_solution))

    worker._process_ipc(progress, None)

    assert len(worker.pool.active_pool) == 1


def test_the_same_slightly_better_peer_is_taken_once_the_search_is_stuck(
        worker, peer_solution):
    worker.stagnation_counter = worker.stag_limit
    progress = shared(best_sol_1=(worker.run_best_cost * 0.99, peer_solution))

    worker._process_ipc(progress, None)

    assert len(worker.pool.active_pool) == 2


def test_the_adventurer_island_ignores_its_peers_until_the_endgame(ctx, peer_cost,
                                                                   peer_solution):
    """Останній острів із чотирьох навмисно шукає окремо.

    Він існує, щоб хтось перевіряв далекі басейни, поки решта сходиться.
    Якби він приймав щеплення з початку, ця роль зникла б.
    """
    local_search = LocalSearch(ctx)
    adventurer = IslandWorker(ctx, KickStrategies(ctx, local_search), local_search,
                              worker_id=3, n_workers=4, max_sims=100_000,
                              beam_width=5, global_archive=[])
    adventurer.ipc_immunity = 0
    adventurer.run_best_sol = [OWN] * ctx.num_pipes
    adventurer.run_best_cost = ctx.get_cached_stats(adventurer.run_best_sol).cost
    adventurer.base_dyn_bonus = 0.0
    adventurer.progress_ratio = 0.5

    adventurer._process_ipc(shared(best_sol_1=(peer_cost, peer_solution)), None)
    assert not adventurer.pool.active_pool, "до 85 % прогону — жодних щеплень"

    adventurer.progress_ratio = 0.9
    adventurer._process_ipc(shared(best_sol_1=(peer_cost, peer_solution)), None)
    assert len(adventurer.pool.active_pool) == 1, "наприкінці прогону — приймає"


def test_an_infeasible_peer_solution_is_refused(worker, ctx):
    """Сусід міг знайти дешевше рішення, яке не тримає тиск.

    Ціна помилки тут вища за звичайну: рішення заходить у пул і стає
    базою для кіків, тобто недопустимість поширюється далі.
    """
    too_cheap = [0] * ctx.num_pipes
    assert ctx.get_cached_stats(too_cheap)[1] < ctx.h_min, "фікстура втратила сенс"

    worker._process_ipc(shared(best_sol_1=(1.0, too_cheap)), None)

    assert len(worker.pool.active_pool) == 1


def test_a_worker_does_not_read_its_own_slot(worker, peer_cost, peer_solution):
    """`best_sol_0` — це запис самого воркера; прочитати його = щепити себе собою."""
    worker._process_ipc(shared(best_sol_0=(peer_cost, peer_solution)), None)

    assert len(worker.pool.active_pool) == 1


# --- порятунок відсталого острова --------------------------------------

def test_a_badly_lagging_worker_adopts_the_global_best(worker, peer_cost,
                                                       peer_solution):
    """Відставання понад 5 % — це вже не «інший басейн», а мертвий.

    На відміну від IPC, порятунок **замінює рекорд**: острів визнає, що
    його гілка пошуку вичерпана.
    """
    worker.global_best_cost = peer_cost

    worker._check_rescue((peer_cost, peer_solution))

    assert worker.rescue_fired
    assert worker.run_best_sol == peer_solution
    assert worker.run_best_cost == peer_cost
    assert worker.stagnation_counter == 0
    assert worker.pool.active_pool[0][2] == peer_solution, (
        "врятоване рішення мусить стати першим у пулі")


def test_the_rescue_clears_the_path_memory(worker, peer_cost, peer_solution):
    """Табу шляхів стосується покинутого басейну — в новому воно шкідливе."""
    worker.global_best_cost = peer_cost
    worker.pool.kick_tabu_set.add((1, 2, 3))

    worker._check_rescue((peer_cost, peer_solution))

    assert not worker.pool.kick_tabu_set


def test_no_rescue_during_the_immunity_window(worker, peer_cost, peer_solution):
    worker.ipc_immunity = 5
    worker.global_best_cost = peer_cost

    worker._check_rescue((peer_cost, peer_solution))

    assert not worker.rescue_fired
    assert worker.run_best_cost != peer_cost


def test_a_small_lag_is_not_a_reason_to_give_up(worker, peer_solution):
    """Півпроцента відставання — нормальна різниця між островами."""
    worker.global_best_cost = worker.run_best_cost * 0.995
    worker.stagnation_counter = 0

    worker._check_rescue((worker.global_best_cost, peer_solution))

    assert not worker.rescue_fired


def test_a_small_lag_plus_deep_stagnation_does_trigger_a_rescue(worker,
                                                                peer_solution):
    """Два слабкі сигнали разом дають підставу, якої не дає жоден окремо."""
    worker.global_best_cost = worker.run_best_cost * 0.97
    worker.stagnation_counter = worker.stag_limit * 2

    worker._check_rescue((worker.global_best_cost, peer_solution))

    assert worker.rescue_fired


def test_the_endgame_uses_a_stricter_lag_threshold(worker, peer_solution):
    """Наприкінці прогону наздоганяти вже нічим, тому поріг падає з 2 % до 0.5 %."""
    worker.global_best_cost = worker.run_best_cost * 0.99
    worker.stagnation_counter = worker.stag_limit * 2

    worker.is_late_game = False
    worker._check_rescue((worker.global_best_cost, peer_solution))
    assert not worker.rescue_fired

    worker.is_late_game = True
    worker._check_rescue((worker.global_best_cost, peer_solution))
    assert worker.rescue_fired


def test_an_infeasible_global_best_is_not_adopted(worker, ctx):
    """Рекорд сусіда теж перевіряється тиском, і з тієї ж причини."""
    worker.global_best_cost = 1.0
    before = list(worker.run_best_sol)

    worker._check_rescue((1.0, [0] * ctx.num_pipes))

    assert worker.run_best_sol == before


def test_nothing_happens_without_a_global_best(worker):
    """На початку паралельного прогону рекорду ще немає."""
    worker.global_best_cost = worker.run_best_cost * 0.5

    worker._check_rescue(None)

    assert not worker.rescue_fired


# --- мінірестарт -------------------------------------------------------

def test_a_moving_search_is_not_restarted(worker, peer_cost, peer_solution):
    worker.stagnation_counter = 1

    worker._check_mini_restart((peer_cost, peer_solution))

    assert len(worker.pool.active_pool) == 1


def test_deep_stagnation_wipes_the_pool_and_the_tabu(worker, peer_cost,
                                                     peer_solution):
    """Терплячість для малих мереж — 10 × `stag_limit`, тобто 40 раундів."""
    worker.stagnation_counter = worker.stag_limit * 10
    worker.pool.kick_tabu_set.add((1, 2))
    worker.pool.tabu_fingerprints["x"] = 1

    worker._check_mini_restart((peer_cost, peer_solution))

    assert not worker.pool.kick_tabu_set
    assert not worker.pool.tabu_fingerprints
    assert all(entry[2] != [OWN] * 6 for entry in worker.pool.active_pool), (
        "старий вміст пулу мусить зникнути")


def test_the_restart_seeds_the_pool_from_the_global_best(worker, peer_cost,
                                                         peer_solution):
    """Рестарт не випадковий: він збурює найкраще відоме рішення.

    Інакше острів, який щойно визнав свою гілку мертвою, почав би з
    порожнього місця й витратив би решту бюджету на повторний спуск.
    """
    worker.stagnation_counter = worker.stag_limit * 10

    worker._check_mini_restart((peer_cost, peer_solution))

    assert worker.pool.active_pool, "пул мусить отримати свіжі зерна"
    assert worker.stagnation_counter == 0


def test_the_endgame_restarts_twice_as_early(worker, peer_cost, peer_solution):
    """`max(4, patience // 2)`: наприкінці прогону чекати 40 раундів пізно."""
    worker.stagnation_counter = worker.stag_limit * 5
    worker.is_late_game = False

    worker._check_mini_restart((peer_cost, peer_solution))
    assert worker.stagnation_counter == worker.stag_limit * 5, "рано для рестарту"

    worker.is_late_game = True
    worker._check_mini_restart((peer_cost, peer_solution))
    assert worker.stagnation_counter == 0


def test_a_restart_without_a_global_best_still_clears_the_pool(worker):
    """Пул чиститься до перевірки `gb` — це і є «м'який рестарт»."""
    worker.stagnation_counter = worker.stag_limit * 10

    worker._check_mini_restart(None)

    assert not worker.pool.active_pool


# --- мікрооптимізація обміном ------------------------------------------

@pytest.fixture
def cheaper(ctx, peer_solution):
    """Те, що «знайшов» `swap_search`: дешевше й допустиме."""
    return list(peer_solution)


def test_the_swap_runs_only_every_eighth_round(worker, cheaper, monkeypatch):
    """Обмін коштує повний прохід по трубах — щороунду це задорого."""
    calls = []
    monkeypatch.setattr(worker.ls, "swap_search",
                        lambda sol, bonus: calls.append(1) or list(sol))

    for round_idx in range(0, 9):
        worker._apply_swap(round_idx, None)

    assert len(calls) == 1, "раунд 0 не рахується, спрацьовує лише 8-й"


def test_a_cheaper_swap_becomes_the_new_record(worker, cheaper, monkeypatch):
    monkeypatch.setattr(worker.ls, "swap_search", lambda sol, bonus: list(cheaper))
    worker.stagnation_counter = 5

    worker._apply_swap(8, None)

    assert worker.run_best_sol == cheaper
    assert worker.stagnation_counter == 4, "виграш зменшує застій на одиницю"
    assert worker.pool.active_pool[0][2] == cheaper


def test_a_swap_that_does_not_help_changes_nothing(worker, monkeypatch):
    monkeypatch.setattr(worker.ls, "swap_search",
                        lambda sol, bonus: [3] * len(sol))
    before = list(worker.run_best_sol)

    worker._apply_swap(8, None)

    assert worker.run_best_sol == before


def test_an_illusory_swap_gain_is_blocked(worker, cheaper, monkeypatch):
    """Щит від ілюзій: виграш понад 2 % перевіряється повторною симуляцією.

    Кеш стохастичних симуляцій може віддати занижену вартість; без цієї
    перевірки воркер прийняв би її як рекорд і далі шукав би навколо
    неіснуючої точки.
    """
    monkeypatch.setattr(worker.ls, "swap_search", lambda sol, bonus: list(cheaper))
    monkeypatch.setattr(worker.ctx, "is_ghost_solution", lambda sol: True)
    before = list(worker.run_best_sol)

    worker._apply_swap(8, None)

    assert worker.run_best_sol == before


def test_a_successful_swap_is_published_to_the_other_islands(worker, cheaper,
                                                             monkeypatch):
    """Без публікації сусіди не дізнаються про знахідку до наступної міграції."""
    monkeypatch.setattr(worker.ls, "swap_search", lambda sol, bonus: list(cheaper))
    progress = shared()

    worker._apply_swap(8, progress)

    assert progress["best_sol_0"][1] == cheaper
    assert progress["global_best"][1] == cheaper


# --- просторове схрещування --------------------------------------------

def looped_worker(looped_ctx):
    local_search = LocalSearch(looped_ctx)
    w = IslandWorker(looped_ctx, KickStrategies(looped_ctx, local_search),
                     local_search, worker_id=0, n_workers=2, max_sims=100_000,
                     beam_width=5, global_archive=[])
    w.run_best_sol = [OWN] * looped_ctx.num_pipes
    w.run_best_cost = looped_ctx.get_cached_stats(w.run_best_sol).cost
    w.base_dyn_bonus = 0.0
    return w


def test_the_crossover_transfers_a_connected_piece_of_the_network(looped_ctx):
    """Дефект: «просторове» схрещування було випадковим.

    Номер труби зберігається в `ctx.edge_to_pipe`, а не на ребрі графа —
    `base_G_flow` будується як `add_edge(u, v)` без атрибутів. Стара
    гілка для звичайного графа шукала ціле число серед `data.values()`,
    тобто серед **порожнього** словника, і не знаходила жодної труби
    ніколи. `local_pipes` лишався порожнім, спрацьовував запасний
    варіант — `random.sample` на 15 % мережі, — і донорські труби
    приходили розкидано по всій мережі замість одного району.

    Тут перевіряється суть: перенесені труби мусять утворювати зв'язний
    шматок навколо епіцентру, а не випадкову вибірку.
    """
    import networkx as nx

    worker = looped_worker(looped_ctx)
    base = [0] * looped_ctx.num_pipes
    donor = [3] * looped_ctx.num_pipes

    # Епіцентр фіксується, щоб очікуваний район можна було порахувати
    # незалежно від кіка — інакше перевірка звелася б до «щось перенесли».
    monkeypatched = "G"
    graph = looped_ctx.base_G_flow
    within = set(nx.single_source_shortest_path_length(
        graph, monkeypatched, cutoff=2).keys())
    expected = {pipe for (u, v), pipe in looped_ctx.edge_to_pipe.items()
                if u in within or v in within}

    original_choice = random.choice
    random.choice = lambda seq: monkeypatched
    try:
        crossed = worker._spatial_crossover(base, donor, T=0.0)
    finally:
        random.choice = original_choice

    moved = {i for i, value in enumerate(crossed) if value == donor[i]}

    assert moved == expected, (
        f"перенесено {sorted(moved)}, а район навколо {monkeypatched} — "
        f"{sorted(expected)}. Розбіжність означає, що спрацював запасний "
        "варіант (випадкові 15 %), тобто труби за ребром не знайшлись.")
    assert len(expected) < looped_ctx.num_pipes, (
        "радіус накрив усю мережу — перевірка б нічого не розрізняла")


def test_a_larger_radius_transfers_more_of_the_network(looped_ctx):
    """`radius = 2 + 6*T` мусить щось означати.

    Доти це був мертвий обчислений вираз: розмір переносу задавався
    запасним варіантом і від T не залежав узагалі.
    """
    worker = looped_worker(looped_ctx)
    base = [0] * looped_ctx.num_pipes
    donor = [3] * looped_ctx.num_pipes

    def moved_count(temperature):
        total = 0
        for seed in range(10):
            random.seed(seed)
            crossed = worker._spatial_crossover(base, donor, temperature)
            total += sum(1 for i, v in enumerate(crossed) if v == donor[i])
        return total / 10

    assert moved_count(1.0) >= moved_count(0.0)


def test_the_crossover_keeps_everything_outside_the_district(looped_ctx):
    """Схрещування — не заміна рішення: поза районом лишається своє."""
    worker = looped_worker(looped_ctx)
    base = [1] * looped_ctx.num_pipes
    donor = [3] * looped_ctx.num_pipes

    random.seed(0)
    crossed = worker._spatial_crossover(base, donor, T=0.0)

    assert any(value == 1 for value in crossed), (
        "перенесено всю мережу — це вже не схрещування, а копіювання")


# --- обмін, який може обірватись ----------------------------------------
#
# `shared_progress` — це `Manager().dict()` у чужому процесі. Запис у
# нього падає з причин, що не стосуються пошуку: менеджер завершився,
# зв'язок обірвався, головний процес зупиняється. З п'яти місць обміну
# три не мали захисту, і падіння в них убивало острів разом із усім, що
# він знайшов; оркестратор бачив лише `[Error] Worker N crashed`.

class Unreachable(dict):
    """Спільний словник, зв'язок із яким обірвано."""

    def __setitem__(self, key, value):
        raise BrokenPipeError("менеджер недоступний")

    def get(self, key, default=None):
        raise EOFError("менеджер недоступний")


def test_a_failed_publication_does_not_stop_the_worker(worker, capsys):
    worker.run_best_cost = 1.0
    worker.run_best_sol = [OWN] * 6

    worker._update_global_best(Unreachable())

    assert "[IPC]" in capsys.readouterr().out


def test_a_record_that_could_not_be_published_is_retried(worker):
    """Найтонше місце виправлення, і його легко зробити неправильно.

    `last_published_cost` означає «сусіди це вже бачили». Якщо посунути
    його після **невдалого** запису, воркер вважатиме рекорд
    опублікованим — і один обрив зв'язку приховає найкраще рішення
    острова до кінця прогону, навіть коли зв'язок відновиться.
    """
    worker.run_best_cost = 1.0
    worker.run_best_sol = [OWN] * 6

    worker._update_global_best(Unreachable())
    assert worker.last_published_cost == float("inf"), (
        "невдалий запис не має рахуватись публікацією")

    reachable = shared()
    worker._update_global_best(reachable)

    assert worker.last_published_cost == 1.0
    assert reachable["best_sol_0"][0] == 1.0


def test_a_sequential_run_still_marks_the_record_as_published(worker):
    """Без островів публікувати нікуди — і це не невдача.

    Інакше послідовний прогін щороунду намагався б «опублікувати»
    рекорд наново.
    """
    worker.run_best_cost = 1.0
    worker.run_best_sol = [OWN] * 6

    worker._update_global_best(None)

    assert worker.last_published_cost == 1.0
