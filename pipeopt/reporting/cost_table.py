"""Таблиця цін: розбір `costs.csv` і заповнення нею конфігурації.

Модуль звався `ga_data` — назва настільки широка, що під нею могло б
лежати будь-що, тоді як усередині одна тема: таблиця типорозмірів і цін.
Індекс у векторі рішення — це позиція в **відсортованому** списку
діаметрів, який повертає саме цей розбір, тому він є контрактом між
солвером, звітами й інтерфейсом (§27.3).
"""

import os
import sys

import pandas as pd

#: Назви стовпців, під якими може бути діаметр і ціна — точний збіг.
DIAMETER_COLUMNS = ("Diameter", "Diam", "D", "diameter", "size")
COST_COLUMNS = ("Cost", "cost", "Price", "price", "UnitCost")
#: Підрядки, за якими стовпець шукається, якщо точного збігу не було.
DIAMETER_HINTS = ("diam", "діаметр")
COST_HINTS = ("cost", "price", "ціна", "вартість")
#: Спроби розбору за порядком: кома, `;`, автовизначення.
SEPARATOR_ATTEMPTS = ({}, {"sep": ";"}, {"sep": None, "engine": "python"})


def _pick_column(frame, names, hints):
    """Стовпець за точною назвою, а якщо такої немає — за підрядком.

    Двоступеневість тут не косметична. `load_cost_table` шукав точний збіг і
    не бачив `Diameter (inch)`; читач інтерфейсу шукав підрядок і не бачив
    `size`. Кожен приймав те, чого не приймав інший, тому об'єднання —
    єдиний варіант, що не забирає нічого в жодного (§27.3).
    """
    exact = next((c for c in names if c in frame.columns), None)
    if exact is not None:
        return exact

    for column in frame.columns:
        low = str(column).lower()
        if any(hint in low for hint in hints):
            return column
    return None


def _read_any_separator(path):
    """Прочитати CSV, пробуючи розділювачі за порядком `SEPARATOR_ATTEMPTS`.

    Ознака невдалого розділювача — один стовпець, а не виняток: `read_csv`
    на файлі з табуляціями не падає, а повертає таблицю з одного стовпця.
    Через це автовизначення, задумане як третя спроба, було недосяжне —
    воно стояло в `except` (§27.4).
    """
    frame = None
    for options in SEPARATOR_ATTEMPTS:
        try:
            frame = pd.read_csv(path, **options)
        except Exception:
            continue
        if len(frame.columns) >= 2:
            return frame

    if frame is None:
        raise ValueError(f"Cannot parse cost table '{path}'")
    return frame


def read_cost_table(path):
    """Прочитати таблицю цін і повернути її, відсортовану за діаметром.

    Повертає `(frame, diameter_column, cost_column)`. Кидає виняток, якщо
    стовпців діаметра чи ціни немає.

    Винесено з `load_cost_table`, бо той самий розбір потрібен інтерфейсу:
    жива карта мережі малюється до створення конфігу, і GUI мав для цього
    власний спрощений читач (`_get_actual_diams`), який не знав ні `;`, ні
    сортування за діаметром (§27.3).
    """
    frame = _read_any_separator(path)

    frame.columns = frame.columns.str.strip()
    diam_col = _pick_column(frame, DIAMETER_COLUMNS, DIAMETER_HINTS)
    cost_col = _pick_column(frame, COST_COLUMNS, COST_HINTS)
    if not diam_col or not cost_col:
        raise KeyError("Columns Diameter/Cost not found")

    return frame.sort_values(by=diam_col), diam_col, cost_col


def diameter_table(path):
    """Діаметри з таблиці цін, або `None`, якщо прочитати не вдалось.

    М'яка обгортка для інтерфейсу: там відсутня чи побита таблиця не
    мусить валити вікно, лише лишає карту без справжніх діаметрів.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        frame, diam_col, _ = read_cost_table(path)
        return frame[diam_col].astype(float).tolist()
    except Exception:
        return None


def load_cost_table( config, silent=False):
    if not os.path.exists(config.cost_file):
        if not silent: print(f"[ERROR] Cost file '{config.cost_file}' not found.")
        sys.exit(1)

    try:
        df, diam_col, cost_col = read_cost_table(config.cost_file)
        config.diameters_raw = df[diam_col].tolist()

        if config.unit_system == "mm":
            config.diameters_m = [d / 1000.0 for d in df[diam_col]]
            if not silent: print("[Config] Units: Millimeters (converted to meters / 1000)")
        else:
            config.diameters_m = [d * 0.0254 for d in df[diam_col]]
            if not silent: print("[Config] Units: Inches (converted to meters * 0.0254)")

        config.costs = dict(zip(df[diam_col], df[cost_col]))

        if not silent: print(f"[Config] Loaded {len(config.diameters_raw)} pipe types.")

    except Exception as e:
        if not silent: print(f"[ERROR] Failed to load costs: {e}")
        sys.exit(1)
