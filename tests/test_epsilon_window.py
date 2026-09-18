"""Допустимість кандидатів променя — інваріант, що лишився від видаленого механізму.

До 15.09 `_beam_search_and_update` містив гілку, яка мала пускати в пул
рішення з **легким** порушенням тиску (надлишок від −0.5 м до 0) з
окремою формулою штрафу:

    is_strictly_valid = p_surplus >= 0.0
    is_epsilon_valid  = p_surplus >= EPSILON_PRESSURE_WINDOW   # -0.5
    score = real_cost + abs(p_surplus) * dyn_bonus * 50.0      # ← ніколи

Досягти її було неможливо, і це доводилось трьома фільтрами, кожен з
яких відсікає недопустимі рішення **до** неї:

1. `LocalSearch.evaluate_candidate` — єдине джерело кандидатів для
   `_generate_mutations` — завершується перевіркою
   `if p_surplus < 0: return ..., None`, а `_generate_mutations` додає
   в `next_gen` лише ті, де рішення не `None`;
2. `LocalSearch.gradient_squeeze` приймає хід лише за `p_val >= h_min`,
   тобто з допустимого рішення не може вийти в недопустиме;
3. у `_beam_search_and_update` кандидат або лишається як є (rank >= 3),
   або проходить `gradient_squeeze` (rank < 3) — в обох випадках
   допустимість зберігається.

Отже надлишок завжди >= 0, а гілка `else` була мертва.

**Заміряно, а не лише доведено.** Лічильники на повних прогонах:

| мережа | кандидатів у промені | з надлишком < 0 | мінімальний надлишок |
|---|---|---|---|
| Ханої, seed 42, 120 тис. | 1588 | **0** | +0.00019 |
| Ханої, seed 7, 120 тис. | 1723 | **0** | +0.00047 |
| Ханої, seed 1, 120 тис. | 1579 | **0** | +0.00047 |
| Балерма, seed 42, 54 тис. | 16 | **0** | +0.000018 |

Цікаве не лише «нуль»: 91 % кандидатів на Ханої мають надлишок у діапазоні
[0, 0.5) — промінь працює впритул до межі тиску. Тобто механізм задумували
для режиму, в якому пошук справді перебуває; недосяжний він через фільтри,
а не через те, що ситуація не трапляється.

**Це має значення поза кодом:** `readme.md` §2 описує «Relaxed-feasibility
(epsilon window у beam search)» як робочий механізм, із формулою штрафу.
Формула не виконувалась жодного разу.

**Чим це закінчилось.** Механізм оживили під прапорцем і зміряли на
трьох мережах (CHANGELOG.md, 15.09): на Фоссі й Балермі він не
спрацьовує взагалі, на Ханої спрацьовує, але виграшу не дає (20 зерен,
знаковий тест p = 0.69). Гілку, константу `EPSILON_PRESSURE_WINDOW` і
розділ readme.md §2 видалено; натомість з'явився лічильник
`LocalSearch.tightness`, який робить тісноту обмеження спостережуваною.

Тести нижче лишились, бо фіксують **інваріанти**, а не видалений код:
обидва фільтри варті гарантії самі по собі — без них у промінь і в пул
потрапляли б недопустимі рішення. Якщо колись хтось свідомо зніме
фільтр, саме ці тести впадуть і скажуть, де прочитати, чим це вже
закінчилось.
"""

import random

import pytest

from pipeopt.solver.local_search import LocalSearch
from pipeopt.solver.island import IslandWorker


@pytest.fixture
def ls(ctx):
    random.seed(3)
    return LocalSearch(ctx)


def surplus(ctx, solution):
    st = ctx.get_cached_stats(solution)
    return st.surplus(ctx.h_min)


# --- фільтр 1: кандидат ніколи не буває недопустимим ------------------

@pytest.mark.parametrize("mode", ["upgrade", "downgrade"])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
def test_evaluate_candidate_never_returns_an_infeasible_solution(ctx, ls, mode, start):
    base = [start] * ctx.num_pipes
    for pipe in range(ctx.num_pipes):
        _, _, solution = ls.evaluate_candidate(base, [pipe], mode, dyn_bonus=1.0)
        if solution is not None:
            assert surplus(ctx, solution) >= 0.0, (
                f"{mode} труби {pipe}: кандидат із надлишком "
                f"{surplus(ctx, solution)} потрапив у промінь")


def test_an_infeasible_candidate_is_reported_as_no_solution(ctx, ls):
    """Коли зменшення діаметра ламає тиск, кандидата немає взагалі."""
    base = [1] + [0] * (ctx.num_pipes - 1)
    _, _, solution = ls.evaluate_candidate(base, [0], "downgrade", dyn_bonus=1.0)
    if solution is None:
        return  # саме те, що перевіряємо
    assert surplus(ctx, solution) >= 0.0


# --- фільтр 2: спуск не виводить із допустимої області ----------------

@pytest.mark.parametrize("start", [1, 2, 3])
@pytest.mark.parametrize("quick", [True, False])
def test_gradient_squeeze_keeps_a_feasible_solution_feasible(ctx, ls, start, quick):
    base = [start] * ctx.num_pipes
    if surplus(ctx, base) < 0:
        pytest.skip("вихідне рішення й так недопустиме")

    squeezed = ls.gradient_squeeze(base, max_passes=5, quick_mode=quick, dyn_bonus=1.0)

    assert surplus(ctx, squeezed) >= 0.0, (
        "спуск приймає хід лише за p >= h_min, тому вийти з допустимої "
        "області він не може")


# --- наслідок: гілка epsilon недосяжна --------------------------------

def test_the_epsilon_branch_is_unreachable_through_the_normal_path(ctx, ls):
    """Наскрізна перевірка ланцюжка, а не окремих ланок.

    Повторює те, що робить `_generate_mutations` + `_beam_search_and_update`
    для кожного рангу, і дивиться на надлишок, який побачила б гілка.
    """
    base = [2] * ctx.num_pipes
    candidates = []
    for pipe in range(ctx.num_pipes):
        for mode in ("upgrade", "downgrade"):
            _, _, solution = ls.evaluate_candidate(base, [pipe], mode, dyn_bonus=1.0)
            if solution is not None:
                candidates.append(solution)

    assert candidates, "передумова: сітка має давати кандидатів"

    seen = []
    for rank, candidate in enumerate(candidates):
        if rank == 0:
            refined = ls.gradient_squeeze(candidate, max_passes=5, quick_mode=False,
                                          dyn_bonus=1.0)
        elif rank < 3:
            refined = ls.gradient_squeeze(candidate, max_passes=3, quick_mode=True,
                                          dyn_bonus=1.0)
        else:
            refined = candidate
        seen.append(surplus(ctx, refined))

    below_zero = [x for x in seen if x < 0.0]
    assert not below_zero, (
        f"{len(below_zero)} із {len(seen)} кандидатів мають надлишок < 0 — "
        "гілка epsilon стала досяжною. Це не помилка тесту: перевірте, чи "
        "оновлено readme.md §2 і чи протестована формула штрафу "
        "`cost + |surplus| * dyn_bonus * 50`, яка досі не виконувалась жодного разу.")

    assert min(seen) >= 0.0


def test_the_deleted_window_left_no_constant_behind(ctx):
    """Регресія: механізму немає — і його сліди теж мають зникнути."""
    assert not hasattr(IslandWorker, "EPSILON_PRESSURE_WINDOW"), (
        "константу видаленого механізму лишили в коді")


# --- те, що прийшло на заміну: замір, а не механізм -------------------

def test_the_tightness_counter_sees_every_candidate(ctx, ls):
    base = [2] * ctx.num_pipes
    for pipe in range(ctx.num_pipes):
        ls.evaluate_candidate(base, [pipe], "downgrade", dyn_bonus=1.0)

    assert ls.tightness.evaluated == ctx.num_pipes


def test_a_network_without_infeasible_candidates_reports_zero(ctx, ls):
    """Так поводиться Фосс: недопустимих кандидатів немає взагалі."""
    # Підвищення діаметра тиск не ламає, тому недопустимих тут бути не може.
    base = [1] * ctx.num_pipes
    for pipe in range(ctx.num_pipes):
        ls.evaluate_candidate(base, [pipe], "upgrade", dyn_bonus=1.0)

    assert ls.tightness.evaluated == ctx.num_pipes, (
        "лічильник рахує кандидатів, які дійшли до перевірки тиску")
    assert ls.tightness.infeasible == 0
    assert ls.tightness.infeasible_share == 0.0
    assert ls.tightness.smallest_deficit is None
    assert ls.tightness.histogram() == {}


def test_the_counter_records_how_deep_the_misses_are(ctx, ls):
    t = ls.tightness
    for surplus in (1.0, 0.0, -0.03, -0.4, -7.0):
        t.note(surplus)

    assert t.evaluated == 5
    assert t.infeasible == 3, "надлишок рівно 0 — це допустимо, не промах"
    assert t.smallest_deficit == pytest.approx(0.03)
    histogram = t.histogram()
    assert histogram["<0.05"] == pytest.approx(1 / 3)
    assert histogram["<0.5"] == pytest.approx(1 / 3)
    assert histogram[">=5"] == pytest.approx(1 / 3), (
        "остання смуга — все, що глибше за найбільшу межу")


def test_the_histogram_is_a_share_of_the_infeasible_not_of_everything(ctx, ls):
    """Частка рахується від недопустимих — інакше вона нічого не каже.

    Саме ця величина відрізняє мережі: на Балермі 100 % недоборів лежать
    під 0.5 м, на Ханої — лише третина.
    """
    t = ls.tightness
    for surplus in [5.0] * 97 + [-0.01, -0.02, -10.0]:
        t.note(surplus)

    assert t.infeasible_share == pytest.approx(0.03)
    assert t.histogram()["<0.05"] == pytest.approx(2 / 3)


def test_the_counter_costs_no_simulations(ctx, ls):
    """Недобір уже обчислений — замір не має додавати жодного виклику."""
    base = [2] * ctx.num_pipes
    ls.evaluate_candidate(base, [0], "downgrade", dyn_bonus=1.0)
    before = ctx.sim_count

    for surplus in (-0.1, 0.5, -2.0):
        ls.tightness.note(surplus)

    assert ctx.sim_count == before


def test_an_empty_counter_does_not_pretend_to_know_the_share(ls):
    """Нуль оцінених — це «немає даних», а не «нуль відсотків»."""
    assert ls.tightness.infeasible_share is None


def test_the_snapshot_carries_everything_a_report_needs(ctx, ls):
    ls.tightness.note(-0.2)
    ls.tightness.note(1.0)

    snapshot = ls.tightness.snapshot()

    assert set(snapshot) == {"evaluated", "infeasible", "infeasible_share",
                             "smallest_deficit", "histogram"}
    assert snapshot["evaluated"] == 2
    assert snapshot["smallest_deficit"] == pytest.approx(0.2)
