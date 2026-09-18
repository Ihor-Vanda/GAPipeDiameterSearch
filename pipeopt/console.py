"""Вивід у консоль: глушіння шуму, роздвоєння `stdout`, формат часу.

Модуль звався `ga_utils` — префікс від генетичного алгоритму, якого в
проєкті давно немає, і назва, яка нічого не забороняє: у `utils` за
півроку лежало б усе, чому не знайшлося місця. Ці три речі насправді про
одне — що саме користувач бачить у консолі: `silence_warnings` глушить
`wntr` і попередження Python, `DualLogger` роздвоює потік у файл прогону,
`format_time` форматує тривалість для виводу.
"""

import logging
import os
import sys
import warnings


def silence_warnings():
    """Прибрати шум із консолі: попередження Python і журнал `wntr`.

    Фільтр ставиться лише тоді, коли користувач не задав `-W` сам, —
    інакше його вибір був би тихо перекритий. Рівень `wntr` знижується
    завжди: він друкує попередження на кожен розбір `.inp`.
    """
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    logging.getLogger('wntr').setLevel(logging.CRITICAL)

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    if seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

class DualLogger:
    """Пише і в термінал, і у файл прогону; ставиться замість `sys.stdout`.

    Каталог створює сам — раніше це робив і виклик у `main.py`, і сам
    логер, а `os.path.dirname("run.txt")` дає порожній рядок, на якому
    `makedirs` падає. Тобто працювало лише тому, що єдине місце виклику
    передавало шлях із каталогом (§30.2).

    **Термінал може не вміти того, що вміє файл.** Файл відкривається в
    UTF-8 і приймає будь-який символ; консоль Windows працює в кодуванні
    локалі (в Україні це cp1251), а перенаправлений `stdout` бере те саме
    кодування навіть тоді, коли інтерактивна консоль була б UTF-8. У
    логах солвера є емодзі (`⚡`, `💎`, `🧬`, `📡`), і `cp1251` їх не має:
    `self.terminal.write` кидав `UnicodeEncodeError` на **першому ж
    рядку** прогону. Далі гірше — `runner.run` ловив виняток і викликав
    `traceback.print_exc()`, який пише туди ж і падає з тієї самої
    причини, ховаючи первинну помилку за вторинною.

    Знайдено 15.09.2026 прогоном на Windows (Python 3.13, cp1251):
    `python tools/reference_run.py --save` не дав жодного результату, а
    `pytest` показав чотири падіння перевірки островів — усі з одного
    кореня, бо обидві перевірки запускають `main.py` підпроцесом і
    перехоплюють його вивід.
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        # Заміна символів, яких немає в кодуванні терміналу, на «?».
        # Кодування не міняємо: cp1251 добре друкує українську, і
        # переводити її в UTF-8 на консолі з іншою кодовою сторінкою
        # означало б виправити емодзі ціною всього решти тексту.
        try:
            self.terminal.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            # Потік без `reconfigure` (підміна в тестах, дуже старий
            # Python) — тоді працює запасний варіант у `write`.
            pass
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, "encoding", None) or "ascii"
            self.terminal.write(
                message.encode(encoding, "replace").decode(encoding))
        # Файл отримує повідомлення **цілим**: він у UTF-8, і втрачати в
        # ньому символи лише тому, що консоль їх не показує, не можна —
        # саме цей файл потім читають при розборі прогону.
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
