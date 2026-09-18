"""`RUNNING.md` не має права розійтися з `main.py`.

Інструкція запуску застаріває тихо: прапорець додали — у документі його
немає; прапорець прибрали — документ обіцяє те, чого не існує. Обидва
випадки виявляються не тоді, коли ламаються, а тоді, коли хтось за
документом запускає прогін і не розуміє, чому не працює.

Автоматично перевіряється не повнота пояснень (це неможливо), а дві
речі, які автоматизуються: **множина прапорців збігається в обидві
сторони**, і значення за замовчуванням у таблиці — ті самі, що в коді.
"""

import re
from pathlib import Path

import pytest

from main import build_parser

DOC = Path(__file__).resolve().parents[1] / "RUNNING.md"


@pytest.fixture(scope="module")
def doc_flags():
    """Прапорці з таблиці параметрів — саме з неї, а не з усього тексту.

    Таблиця знаходиться за заголовком: інакше перевірка ловила б будь-яку
    згадку `--seed` у прикладі команди й мовчки проходила б, навіть коли
    рядка в таблиці немає.
    """
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## Усі параметри")
    end = text.index("### `--units`")
    table = text[start:end]
    return set(re.findall(r"^\| `(--[a-z_]+)`", table, flags=re.MULTILINE))


@pytest.fixture(scope="module")
def parser_flags():
    flags = set()
    for action in build_parser()._actions:
        flags.update(opt for opt in action.option_strings if opt.startswith("--"))
    return flags - {"--help"}


def test_the_document_lists_every_flag(doc_flags, parser_flags):
    missing = parser_flags - doc_flags
    assert not missing, (
        f"прапорці є в коді, але не в RUNNING.md: {sorted(missing)}")


def test_the_document_invents_no_flags(doc_flags, parser_flags):
    extra = doc_flags - parser_flags
    assert not extra, (
        f"RUNNING.md обіцяє прапорці, яких немає: {sorted(extra)}")


def test_the_table_was_actually_found(doc_flags):
    """Самоохорона: якщо заголовок розділу перейменують, таблиця стане
    порожньою — і обидві перевірки вище проходили б завжди."""
    assert len(doc_flags) >= 10, (
        f"у таблиці знайдено лише {len(doc_flags)} прапорців — "
        "схоже, розділ перейменували, і перевірка дивиться не туди")


@pytest.mark.parametrize("flag, default", [
    ("--units", "mm"),
    ("--hmin", 30.0),
    ("--runs", 1),
    ("--cores", 0),
    ("--run_mode", "analytical"),
    ("--v_opt", 1.0),
])
def test_defaults_in_the_table_match_the_code(flag, default):
    """Значення за замовчуванням у таблиці — не прикраса: за ними люди
    вирішують, який прапорець можна не писати."""
    action = next(a for a in build_parser()._actions
                  if flag in a.option_strings)

    assert action.default == default
