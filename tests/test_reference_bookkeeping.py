"""Знімок еталона як файл: що в ньому зберігається і що втрачається.

Числа еталона перевіряє `test_reference.py` — повільно, справжніми
прогонами. Цей файл про інше й швидкий: про **бухгалтерію знімка**.

З 16.09.2026 `tests/reference_baseline.json` комітиться в репозиторій,
і в нього додалось поле `confirmed_on` — перелік середовищ, у яких
`--compare` зійшовся побітово. Це поле і є відповіддю на питання
«наскільки ці числа переносні»: не припущення, а список машин.

Тому в нього з'явилось правило, яке легко зламати непомітно:

* перезняття знімка **не має** стирати підтвердження, якщо числа ті самі
  (перезняли на тій самій поведінці — історія лишається чинною);
* перезняття **має** стирати їх, якщо числа інші (старі підтвердження
  стосувались інших чисел, і залишити їх означало б збрехати про те, що
  нові числа хтось відтворив).

Помилка в будь-якому з цих двох правил тиха: файл виглядає правильним, а
твердження про відтворюваність — хибним.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import reference_run  # noqa: E402

LINUX = {"python": "3.11.15", "platform": "Linux-x86_64", "numba": True}
WINDOWS = {"python": "3.13.0", "platform": "Windows-11", "numba": True}


def snapshot(cost=6.0812):
    return {"environment": LINUX,
            "runs": {"analytical/seed42/40k": {"cost": cost, "sims": 40982}}}


@pytest.fixture
def baseline_file(tmp_path):
    return tmp_path / "reference_baseline.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- запис знімка -------------------------------------------------------

def test_a_fresh_snapshot_has_nothing_confirmed_yet(baseline_file):
    """Щойно зняті числа ніхто ще не відтворював — і файл це показує."""
    reference_run.save(baseline_file, snapshot())

    assert not read(baseline_file).get("confirmed_on")


def test_retaking_the_same_numbers_keeps_the_confirmations(baseline_file):
    """Головне правило: перезняття на тій самій поведінці нічого не втрачає.

    Знімок перезнімають і просто так — наприклад, щоб перевірити, що
    нічого не поїхало. Якби це щоразу стирало список, він ніколи б не
    накопичився, і поле було б декоративним.
    """
    reference_run.save(baseline_file, snapshot())
    reference_run.add_confirmation(baseline_file, read(baseline_file), WINDOWS)

    reference_run.save(baseline_file, snapshot())

    assert read(baseline_file)["confirmed_on"] == [WINDOWS]


def test_retaking_different_numbers_drops_the_confirmations(baseline_file):
    """Зворотне правило, і воно важливіше за перше.

    Підтвердження стосується **конкретних чисел**. Змінились числа —
    жодна з тих машин нових не відтворювала, і список мусить зникнути,
    інакше знімок стверджує неправду.
    """
    reference_run.save(baseline_file, snapshot())
    reference_run.add_confirmation(baseline_file, read(baseline_file), WINDOWS)

    reference_run.save(baseline_file, snapshot(cost=6.0961))

    assert not read(baseline_file).get("confirmed_on")


def test_the_snapshot_is_written_as_readable_json(baseline_file):
    """Файл комітиться, тобто його читають у diff — він має бути читабельним."""
    reference_run.save(baseline_file, snapshot())
    text = baseline_file.read_text(encoding="utf-8")

    assert "\n" in text, "знімок в один рядок не читається в diff"
    assert "Linux" in text


# --- підтвердження ------------------------------------------------------

def test_a_new_environment_is_added(baseline_file):
    reference_run.save(baseline_file, snapshot())

    added = reference_run.add_confirmation(baseline_file, read(baseline_file),
                                           WINDOWS)

    assert added
    assert read(baseline_file)["confirmed_on"] == [WINDOWS]


def test_the_same_environment_is_not_counted_twice(baseline_file):
    """Інакше десять прогонів на одній машині виглядали б як десять машин."""
    reference_run.save(baseline_file, snapshot())
    reference_run.add_confirmation(baseline_file, read(baseline_file), WINDOWS)

    added = reference_run.add_confirmation(baseline_file, read(baseline_file),
                                           WINDOWS)

    assert not added
    assert len(read(baseline_file)["confirmed_on"]) == 1


def test_confirmations_accumulate_in_order(baseline_file):
    reference_run.save(baseline_file, snapshot())
    reference_run.add_confirmation(baseline_file, read(baseline_file), LINUX)
    reference_run.add_confirmation(baseline_file, read(baseline_file), WINDOWS)

    assert read(baseline_file)["confirmed_on"] == [LINUX, WINDOWS]


def test_the_numbers_survive_a_confirmation(baseline_file):
    """Підтвердження дописує факт про середовище — і нічого більше."""
    reference_run.save(baseline_file, snapshot())
    before = read(baseline_file)["runs"]

    reference_run.add_confirmation(baseline_file, read(baseline_file), WINDOWS)

    assert read(baseline_file)["runs"] == before


# --- знімок, який лежить у репозиторії -----------------------------------

def test_the_committed_snapshot_is_valid_json():
    """Файл тепер комітиться, тому зламаний JSON зламав би `pytest` усім."""
    if not reference_run.BASELINE.exists():
        pytest.skip("немає знімка — див. test_reference.py")

    data = read(reference_run.BASELINE)

    assert data["runs"], "знімок без прогонів"
    assert data["environment"]["platform"]


def test_the_committed_snapshot_covers_every_case():
    """Прогін, якого немає у знімку, мовчки пропускається в `test_reference`.

    Тобто новий випадок у `CASES` без перезняття знімка виглядає як
    зелений еталон, не перевіривши нічого.
    """
    if not reference_run.BASELINE.exists():
        pytest.skip("немає знімка")

    recorded = set(read(reference_run.BASELINE)["runs"])
    expected = {name for name, _ in reference_run.CASES}

    assert expected <= recorded, (
        f"у знімку немає прогонів {sorted(expected - recorded)} — "
        "перезніміть його командою `python tools/reference_run.py --save`")
