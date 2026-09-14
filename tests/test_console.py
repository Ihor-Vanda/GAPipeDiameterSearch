"""Вивід у консоль: формат часу, роздвоєння `stdout`, глушіння шуму.

Модуль (раніше `ga_utils`) мав 36 % покриття: перевірялись тільки рядки
імпорту. Це виглядає нешкідливо, бо там тридцять рядків, — але через
`DualLogger` проходить **весь** вивід прогону CLI, включно з тим, що
потім читається у вкладці логів інтерфейсу, а `format_time` формує рядок,
який користувач бачить у звіті.
"""

import os
import sys

import pytest

from console import DualLogger, format_time, silence_warnings


# --- формат тривалості -------------------------------------------------

@pytest.mark.parametrize("seconds,shown", [
    (0.0, "0.0s"),
    (0.04, "0.0s"),
    (7.25, "7.2s"),
    (59.9, "59.9s"),
])
def test_under_a_minute_keeps_one_decimal(seconds, shown):
    assert format_time(seconds) == shown


@pytest.mark.parametrize("seconds,shown", [
    (60, "1m 0s"),
    (90, "1m 30s"),
    (599, "9m 59s"),
    (3599, "59m 59s"),
])
def test_minutes_drop_the_decimal(seconds, shown):
    assert format_time(seconds) == shown


@pytest.mark.parametrize("seconds,shown", [
    (3600, "1h 0m"),
    (5400, "1h 30m"),
    (86399, "23h 59m"),
])
def test_hours_drop_the_seconds(seconds, shown):
    assert format_time(seconds) == shown


def test_the_boundaries_do_not_overlap():
    """59.9 с — ще секунди, 60 с — уже хвилини; 3599 — ще хвилини, 3600 —
    години. Помилка на межі дала б `60.0s` або `0m 3600s`."""
    assert format_time(59.999).endswith("s") and "m" not in format_time(59.999)
    assert format_time(60).startswith("1m")
    assert format_time(3599).startswith("59m")
    assert format_time(3600).startswith("1h")


def test_a_long_run_is_reported_in_hours():
    """Квота XLARGE — 30 млн симуляцій, тобто близько 20 діб на одному
    ядрі: формат мусить витримувати такі числа, а не тільки хвилини."""
    assert format_time(20 * 24 * 3600) == "480h 0m"


# --- роздвоєння виводу -------------------------------------------------

def test_output_goes_to_both_the_terminal_and_the_file(tmp_path, capsys):
    path = tmp_path / "run.log"
    logger = DualLogger(str(path))
    logger.write("рядок прогону\n")
    logger.flush()

    assert path.read_text(encoding="utf-8") == "рядок прогону\n"
    assert "рядок прогону" in capsys.readouterr().out


def test_the_file_is_appended_not_replaced(tmp_path):
    """Лог прогону відкривається в режимі `a`: у GUI один каталог може
    приймати кілька прогонів підряд, і другий не мусить стирати перший."""
    path = tmp_path / "run.log"
    first = DualLogger(str(path))
    first.write("перший\n")
    first.flush()

    second = DualLogger(str(path))
    second.write("другий\n")
    second.flush()

    assert path.read_text(encoding="utf-8") == "перший\nдругий\n"


def test_each_write_reaches_the_file_immediately(tmp_path):
    """`flush` після кожного запису — не дрібниця: інтерфейс читає цей
    файл під час прогону, і без скидання буфера вкладка логів була б
    порожньою до самого кінця."""
    path = tmp_path / "run.log"
    logger = DualLogger(str(path))
    logger.write("живий рядок\n")
    assert "живий рядок" in path.read_text(encoding="utf-8")


def test_a_missing_directory_is_created(tmp_path):
    path = tmp_path / "logs" / "deep" / "run.log"
    DualLogger(str(path)).write("x")
    assert path.exists()


def test_a_bare_filename_is_not_an_error(tmp_path, monkeypatch):
    """`os.path.dirname("run.txt")` — порожній рядок, на якому `makedirs`
    падав. Працювало лише тому, що єдине місце виклику передавало шлях
    із каталогом (§30.2)."""
    monkeypatch.chdir(tmp_path)
    DualLogger("run.log").write("x")
    assert (tmp_path / "run.log").exists()


def test_cyrillic_survives_the_round_trip(tmp_path):
    """Кодування задається явно: без `encoding='utf-8'` на Windows файл
    писався б у cp1251 і лог із українськими рядками бився б."""
    path = tmp_path / "run.log"
    logger = DualLogger(str(path))
    logger.write("Найнижчий тиск: 30.01 м (Вузол: 13)\n")
    logger.flush()
    assert "Найнижчий тиск" in path.read_text(encoding="utf-8")


def test_the_logger_can_stand_in_for_stdout(tmp_path):
    """Саме так його і використовує `main.py`: `sys.stdout = DualLogger(...)`,
    тому потрібні обидва методи файлового об'єкта, якими користується
    `print`."""
    path = tmp_path / "run.log"
    logger = DualLogger(str(path))
    saved = sys.stdout
    try:
        sys.stdout = logger
        print("через print")
    finally:
        sys.stdout = saved
    logger.flush()
    assert "через print" in path.read_text(encoding="utf-8")


# --- глушіння попереджень ----------------------------------------------

def test_warnings_are_silenced_when_the_user_asked_for_nothing(monkeypatch):
    monkeypatch.setattr(sys, "warnoptions", [])
    monkeypatch.delenv("PYTHONWARNINGS", raising=False)

    silence_warnings()
    assert os.environ.get("PYTHONWARNINGS") == "ignore"


def test_an_explicit_dash_w_is_not_overridden(monkeypatch):
    """Якщо користувач задав `-W` сам, його вибір не перекривається —
    інакше `python -W error main.py` тихо нічого не робив би."""
    monkeypatch.setattr(sys, "warnoptions", ["error"])
    monkeypatch.delenv("PYTHONWARNINGS", raising=False)

    silence_warnings()
    assert "PYTHONWARNINGS" not in os.environ


def test_silencing_is_idempotent(monkeypatch):
    """Викликається на імпорті `main.py`, тому мусить витримувати
    повторний виклик."""
    monkeypatch.setattr(sys, "warnoptions", [])
    silence_warnings()
    silence_warnings()
    assert os.environ.get("PYTHONWARNINGS") == "ignore"


def test_silencing_warnings_quiets_wntr():
    import logging

    silence_warnings()
    assert logging.getLogger("wntr").level == logging.CRITICAL
