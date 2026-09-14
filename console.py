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
    elif seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    else: return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
    
class DualLogger(object):
    """Пише і в термінал, і у файл прогону; ставиться замість `sys.stdout`.

    Каталог створює сам — раніше це робив і виклик у `main.py`, і сам
    логер, а `os.path.dirname("run.txt")` дає порожній рядок, на якому
    `makedirs` падає. Тобто працювало лише тому, що єдине місце виклику
    передавало шлях із каталогом (§30.2).
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()