"""Вивід у консоль: глушіння шуму, роздвоєння `stdout`, формат часу.

Модуль звався `ga_utils` — префікс від генетичного алгоритму, якого в
проєкті давно немає, і назва, яка нічого не забороняє: у `utils` за
півроку лежало б усе, чому не знайшлося місця. Ці три речі насправді про
одне — що саме користувач бачить у консолі: `silence_warnings` глушить
`wntr` і попередження Python, `DualLogger` роздвоює потік у файл прогону,
`format_time` форматує тривалість для виводу.
"""

import os
import sys
import warnings


def silence_warnings():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
    
    import logging
    logging.getLogger('wntr').setLevel(logging.CRITICAL)

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    elif seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    else: return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
    
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()