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

from pipeopt.console import DualLogger, format_time, silence_warnings


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

# --- термінал, який не вміє того, що вміє файл -------------------------
#
# Дефект, знайдений прогоном на Windows 15.09.2026. Файл прогону
# відкривається в UTF-8 і приймає будь-який символ, а консоль працює в
# кодуванні локалі — в Україні це cp1251. Емодзі з логів солвера (`⚡`,
# `💎`, `🧬`, `📡`) у cp1251 не існує, тому `terminal.write` кидав
# `UnicodeEncodeError` на першому ж рядку прогону.
#
# Важливо, **коли саме** це спрацьовує. Інтерактивна консоль у Windows
# часто вміє UTF-8, тому руками прогін ішов нормально. Але
# `subprocess.run(..., capture_output=True)` робить stdout трубою, і для
# труби Python бере кодування локалі. Тобто падало саме те, що запускає
# прогін підпроцесом: `tools/reference_run.py` і перевірка островів —
# чотири падіння `pytest` з одного кореня.


class Cp1251Terminal:
    """Консоль, яка вміє українську й не вміє емодзі.

    Саме те, що дає Windows у нашій локалі. `io.TextIOWrapper` з
    `encoding="cp1251"` поводиться так само, тому підміна чесна, а не
    умовна: цей об'єкт кидає справжній `UnicodeEncodeError` з того самого
    кодека.
    """

    encoding = "cp1251"

    def __init__(self, allow_reconfigure=True):
        self.written = []
        self.errors = "strict"
        self._allow_reconfigure = allow_reconfigure

    def reconfigure(self, errors=None, **_):
        if not self._allow_reconfigure:
            raise AttributeError("reconfigure")
        if errors:
            self.errors = errors

    def write(self, message):
        self.written.append(message.encode(self.encoding, self.errors)
                            .decode(self.encoding))

    def flush(self):
        pass


@pytest.fixture
def cp1251(monkeypatch):
    def install(allow_reconfigure=True):
        terminal = Cp1251Terminal(allow_reconfigure)
        monkeypatch.setattr(sys, "stdout", terminal)
        return terminal
    return install


def test_an_emoji_does_not_kill_the_run_on_a_windows_console(tmp_path, cp1251):
    """Головна перевірка: прогін не має падати через символ у лозі.

    Ціна помилки тут непропорційна причині — багатогодинний пошук
    зупиняється на першому ж рядку виводу через прикрасу в повідомленні.
    """
    terminal = cp1251()
    logger = DualLogger(str(tmp_path / "logs" / "run.txt"))

    logger.write("[AnalyticalSolver] ⚡ Initiating Island Model Search...\n")

    assert terminal.written, "у термінал не потрапило нічого"


def test_the_file_keeps_the_emoji_the_console_could_not_show(tmp_path, cp1251):
    """Файл у UTF-8 і мусить лишитись повним.

    Обрізати його під можливості консолі означало б втратити дані в
    єдиному місці, де їх потім читають, — у логу прогону.
    """
    cp1251()
    path = tmp_path / "logs" / "run.txt"
    logger = DualLogger(str(path))

    logger.write("⚡ старт\n")
    logger.log.flush()

    assert "⚡" in path.read_text(encoding="utf-8")


def test_ukrainian_still_prints_correctly(tmp_path, cp1251):
    """cp1251 добре друкує українську — і мусить друкувати далі.

    Тому кодування терміналу не перевизначається на UTF-8: це полагодило
    б емодзі ціною всього решти тексту на консолі з іншою кодовою
    сторінкою. Замінюються лише символи, яких кодування не має.
    """
    terminal = cp1251()
    logger = DualLogger(str(tmp_path / "logs" / "run.txt"))

    logger.write("Задача нерозв'язна: перевірте одиниці\n")

    assert "Задача нерозв'язна" in "".join(terminal.written)


def test_a_terminal_without_reconfigure_is_handled_too(tmp_path, cp1251):
    """Запасний варіант у `write` для потоку, який не вміє `reconfigure`.

    Такий потік — це, наприклад, підміна в тестах або обгортка стороннього
    інструменту. Покладатись лише на `reconfigure` означало б лагодити
    дефект для справжньої консолі й лишати його для всіх обгорток.
    """
    terminal = cp1251(allow_reconfigure=False)
    logger = DualLogger(str(tmp_path / "logs" / "run.txt"))

    logger.write("💎 [SWAP] Micro-Optimization\n")

    assert terminal.written


def test_what_the_console_shows_instead_of_an_emoji(tmp_path, cp1251):
    """Символ замінюється, а не ковтається: рядок лишається читабельним."""
    terminal = cp1251()
    logger = DualLogger(str(tmp_path / "logs" / "run.txt"))

    logger.write("⚡ Initiating\n")

    shown = "".join(terminal.written)
    assert "Initiating" in shown
    assert "⚡" not in shown


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
