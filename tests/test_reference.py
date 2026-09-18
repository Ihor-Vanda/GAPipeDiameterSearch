"""Еталонні прогони як частина `pytest`.

Ідея: критерій «поведінка не змінилась» має спрацьовувати сам, а не тоді,
коли про нього згадали. Тому знімок еталона звіряється звичайним тестом.

**Але це не звичайний тест, і саме тому він влаштований так.** Звичайний
тест каже «правильно / неправильно». Цей каже «так само / інакше», а це
різні речі: зміна числа тут не завжди дефект. У сесії 14.09 еталон
розійшовся, і розходження було **правильним** — виправлення
`_emergency_respawn` мусило змінити траєкторію пошуку. Якби це був
золотий тест зі значенням у коді, найпростішою реакцією було б
«оновити очікуване число», і рішення, яке належить обґрунтувати й
записати, розчинилось би в рефлексі.

Звідси три властивості:

1. **Знімок не комітиться** — але причина виявилась слабшою, ніж тут
   було написано. Доти цей абзац стверджував, що числа залежать від
   машини, збірки Python і стану JIT numba, тому спільного
   «правильного» знімка не існує. Це було **припущення, яке ніхто не
   міряв**.

   Замір 16.09.2026 на двох машинах:

   | | Linux, Python 3.11 | Windows 11, Python 3.13 |
   |---|---|---|
   | `seed42/40k` | 6.0812 / 40 982 | 6.0812 / 40 982 |
   | `seed7/40k` | 6.0974 / 40 868 | 6.0974 / 40 868 |
   | `seed42/120k` | 6.0961 / 121 023 | 6.0961 / 121 023 |
   | `fast/seed42` | 6.8013 / 1 852 | 6.8013 / 1 852 |

   Збіглось усе, включно з лічильником симуляцій — а він чутливіший за
   вартість (див. нижче). Тобто пошук пройшов **тим самим шляхом** на
   різних ОС і різних збірках Python.

   Знімок і далі не комітиться, але тепер це обережність, а не факт:
   дві машини — це дві точки, а не доведення. Інша збірка EPANET або
   інша версія `wntr` цілком можуть дати інше. Документовані числа
   живуть у `CHANGELOG.md`, де їх читає людина.

2. **Немає знімка — тест пропускається** з інструкцією, як його зробити.
   Це не тихе «все добре»: `pytest` друкує причину пропуску.

3. **Є знімок — тест іде завжди.** Опт-ін відбувається один раз, коли ви
   створюєте файл; далі забути неможливо. Ціна — близько хвилини до
   кожного повного прогону тестів. Коли треба швидко:
   `pytest -m "not reference"`.

## Друга половина: прогін на кількох островах

Еталон відтворюваний лише тому, що йде з `--sequential`. Але саме цей
прапорець **вимикає міжострівний шар**: при `shared_progress is None` не
виконуються ані міграція, ані IPC, ані порятунок відсталого воркера.
Тобто зелений еталон нічого про них не каже — і це не теорія: етапи
15–16 переписали міграцію, і жоден еталонний прогін її не торкнувся.

Тому поруч є `test_the_island_layer_still_works`: короткий **паралельний**
прогін, у якому порівнюються не числа (вони не відтворювані), а
структурні факти — усі острови дійшли до кінця, обмін між ними був,
рішення допустиме. Разом ці дві перевірки дають «числа ті самі» **і**
«механізми, яких числа не бачать, живі».

## Чого цей тест НЕ ловить — заміряно, не припущення

Еталон перевіряє **траєкторію пошуку**, а не покриття коду. Три навмисні
зміни, внесені для перевірки самого харнеса:

| зміна | спіймано |
|---|---|
| `RECORD_RESET_FRACTION` 0.005 -> 0.05 | **так** (3 падіння) |
| `GHOST_SUSPICION_FRACTION` 0.02 -> 0.03 | ні |
| `EPSILON_PRESSURE_WINDOW` -0.5 -> -0.4 | ні |

Причину виміряно лічильниками на прогоні `seed=42, 120 тис.`:

* `_is_ghost_gain` викликається **12 разів**, і лише **2** з них
  перевищують поріг підозри. Зсув порогу просто не перевертає жодного з
  цих двох випадків;
* вікно epsilon спрацьовує **0 разів** — на Ханої не трапляється рішення
  з надлишком тиску між −0.5 і 0. Тобто на цій мережі змінити
  `EPSILON_PRESSURE_WINDOW` не може вплинути ні на що.

Висновок практичний: **зелений еталон не означає «код не змінився»**, він
означає «на Ханої, на цих чотирьох прогонах, пошук пройшов тим самим
шляхом». Для гілок, яких Ханої не торкається, потрібні юніт-тести — і
саме тому вони писались окремо (`test_ghost_shield.py`,
`test_emergency_respawn.py`), а не заміняються еталоном.

Заодно це самостійна знахідка: вікно відносної допустимості в променевому
пошуку на Ханої не активується жодного разу. Чи це властивість малих
мереж, чи механізм мертвий взагалі — питання до заміру на Балермі.

## Коли тест упав

Є рівно дві чесні реакції:

* це регресія → виправити код;
* це свідома зміна → записати в `CHANGELOG.md`, **чому саме**, додати
  A/B, і аж тоді перезняти знімок (`python tools/reference_run.py --save`).

Зверніть увагу, який саме тест упав: `..._costs_the_same_number_of_simulations`
чутливіший за `..._gives_the_same_cost`. Дві траєкторії легко приходять в
один оптимум різними шляхами — у заміру вище `seed=7` дав ту саму
вартість і **інший** лічильник симуляцій.

Перезняти знімок, не написавши чому, — єдиний спосіб зробити цей тест
безглуздим.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import reference_run  # noqa: E402

pytestmark = [pytest.mark.reference, pytest.mark.slow]

MISSING_BASELINE = f"""немає знімка еталона.

Файл {reference_run.BASELINE.relative_to(ROOT)} **комітиться** — якщо його
немає, це або неповне клонування, або його видалили руками. Відновіть
його з репозиторію; знімати наново треба лише тоді, коли поведінка
змінена свідомо й описана в `CHANGELOG.md`:

    python tools/reference_run.py --save"""


@pytest.fixture(scope="module")
def baseline():
    if not reference_run.BASELINE.exists():
        pytest.skip(MISSING_BASELINE)
    return json.loads(reference_run.BASELINE.read_text(encoding="utf-8"))


def environment_note(baseline, current):
    """Рядок про розбіжність середовищ — для повідомлення про падіння.

    Не окрема перевірка, і навмисно. Раніше розбіжність у `numba` була
    **твердженням** (`assert`), тобто числа вважались залежними від JIT.
    Замір це не підтвердив: знімок, знятий без numba, збігся побітово з
    прогонами, де numba є. Тому середовище тепер — діагностика при
    падінні, а не причина падіння: воно допомагає зрозуміти, **чому**
    числа поїхали, і не створює хибних падінь, коли вони не поїхали.
    """
    before, after = baseline.get("environment", {}), current.get("environment", {})
    differences = [f"{key}: {before.get(key)!r} -> {after.get(key)!r}"
                   for key in ("python", "platform", "numba")
                   if key in before and before[key] != after.get(key)]
    if not differences:
        return ""
    confirmed = baseline.get("confirmed_on", [])
    return ("\n\nСередовище відрізняється від того, у якому знято знімок:\n  "
            + "\n  ".join(differences)
            + f"\nЗнімок відтворювався на {len(confirmed)} середовищ(ах); "
              "якщо ваше нове — це може бути не регресія, а межа переносності. "
              "Перевірте версію wntr і збірку EPANET перед тим, як шукати "
              "регресію в коді.")


@pytest.fixture(scope="module")
def current(baseline):
    """Один прогін еталона на весь модуль — він коштує близько хвилини."""
    if not reference_run.INP.exists():
        pytest.skip(f"немає {reference_run.INP.relative_to(ROOT)}")
    return reference_run.collect(timeout=1800)


def runs_of(snapshot):
    return snapshot.get("runs", snapshot)


@pytest.mark.parametrize("case", [name for name, _ in reference_run.CASES])
def test_the_run_gives_the_same_cost(baseline, current, case):
    before, after = runs_of(baseline).get(case), runs_of(current)[case]
    if before is None:
        pytest.skip(f"прогону {case!r} не було у знімку — перезніміть його")

    assert after["cost"] == before["cost"], (
        f"{case}: вартість {before['cost']}M$ -> {after['cost']}M$.\n"
        "Якщо це свідома зміна поведінки — опишіть її в CHANGELOG.md з A/B "
        "і перезніміть знімок; якщо ні — це регресія."
        + environment_note(baseline, current))


@pytest.mark.parametrize("case", [name for name, _ in reference_run.CASES])
def test_the_run_costs_the_same_number_of_simulations(baseline, current, case):
    """Вартість може збігтись випадково; кількість симуляцій — ні.

    Дві траєкторії пошуку легко приходять в один локальний оптимум різними
    шляхами. Лічильник симуляцій ловить саме шлях, тому він чутливіший за
    результат — і часто падає перший.
    """
    before, after = runs_of(baseline).get(case), runs_of(current)[case]
    if before is None:
        pytest.skip(f"прогону {case!r} не було у знімку")

    assert after["sims"] == before["sims"], (
        f"{case}: {before['sims']:,} -> {after['sims']:,} симуляцій. "
        "Вартість могла збігтись, але пройдений шлях інший."
        + environment_note(baseline, current))


def test_the_baseline_carries_an_environment_fingerprint(baseline):
    """Без відбитка розходження неможливо діагностувати.

    Сам по собі відбиток нічого не забороняє (див. `environment_note`),
    але коли числа поїхали, перше питання — «а середовище те саме?».
    Знімок без цієї відповіді змушує гадати.
    """
    fingerprint = baseline.get("environment")

    assert fingerprint, "знімок без відбитка середовища — перезніміть його"
    assert set(fingerprint) >= {"python", "platform", "numba"}


def test_the_baseline_records_where_it_was_reproduced(baseline):
    """Знімок комітиться, тому мусить нести докази, а не лише числа.

    `confirmed_on` — список середовищ, у яких `--compare` зійшовся
    побітово. Він і є відповіддю на питання «наскільки ці числа
    переносні»: не припущенням, а переліком машин. Поповнюється командою

        python tools/reference_run.py --compare --confirm
    """
    confirmed = baseline.get("confirmed_on")

    assert confirmed, (
        "знімок не має жодного підтвердження. Якщо числа щойно перезняті — "
        "це нормально: підтвердьте їх на другій машині командою "
        "`python tools/reference_run.py --compare --confirm`.")
    assert all("platform" in entry for entry in confirmed)


# --- міжострівний шар, якого послідовний еталон не бачить --------------

@pytest.fixture(scope="module")
def island_facts():
    """Один короткий паралельний прогін на весь модуль (близько 25 с)."""
    if not reference_run.INP.exists():
        pytest.skip(f"немає {reference_run.INP.relative_to(ROOT)}")
    return reference_run.island_check(timeout=1800)


def test_the_island_layer_still_works(island_facts):
    """Наскрізна перевірка: що саме зламалось, каже `island_problems`."""
    problems = reference_run.island_problems(island_facts)

    assert not problems, (
        "паралельний прогін виявив проблеми:\n  - " + "\n  - ".join(problems)
        + "\n(числа тут не порівнюються — лише структурні факти)")


def test_every_island_ran_to_the_end(island_facts):
    """Воркер, що впав посередині, не друкує підсумковий рядок.

    Пул ковтає виняток у воркері й друкує `[Error] Worker N crashed`, тому
    прогін із мертвим островом виглядає успішним: решта островів дадуть
    результат. Ця перевірка — єдине, що це ловить.
    """
    assert island_facts["islands_finished"] == island_facts["islands_expected"]


def test_the_islands_actually_talked_to_each_other(island_facts):
    """Нуль подій обміну = перевірка нічого не перевірила.

    Міграція, IPC і порятунок — єдине, що відрізняє острівну модель від
    N незалежних пошуків. Якщо жодна з них не спрацювала, то паралельний
    прогін пройшов тими самими шляхами, що й послідовний, і ця перевірка
    дублює еталон замість доповнювати його.
    """
    assert island_facts["cross_island_events"] > 0


def test_the_parallel_run_returns_a_feasible_solution(island_facts):
    assert island_facts["feasible"]
    assert island_facts["cost"] is not None
