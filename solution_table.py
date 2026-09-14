"""Розбір і підготовка `solution.csv` до показу — без жодного віджета.

Файл рішення читали двоє: `netdraw.diameters_from_table` (щоб намалювати
карту) і перегляд результатів (щоб показати таблицю). Обидва шукали
стовпець діаметра власним циклом по тому самому списку підказок, і обидва
самі перекладали індекс типорозміру в діаметр — двічі один контракт, у
двох файлах, один з яких тягне tkinter і тому не перевірявся (§27.5).

Тут це один раз і без інтерфейсу, тому перевіряється тестами.

Головна тонкість — у стовпці рішення може стояти або сам діаметр, або
індекс у таблиці типорозмірів. Розрізняються вони тільки тим, що індекс —
ціле число в межах таблиці. Тобто `12` у файлі з таблицею Ханою — це
індекс 12, якого немає (таблиця з шести рядків), а `4` — це четвертий
типорозмір, 30 дюймів, а не діаметр 4. Правило вибору тут одне для карти
й для таблиці; раніше воно було двома окремими копіями.
"""

import pandas as pd

#: Назви стовпців, у яких шукається діаметр у `solution.csv`.
DIAMETER_COLUMN_HINTS = ("diam", "діаметр", "d_")
#: Стовпці, які показуються як назви, а не як числа.
NAME_COLUMN_HINTS = ("id", "node", "вузол")


def diameter_column(frame):
    """Назва стовпця з діаметром; якщо підказка не спрацювала — останній.

    Останній стовпець як запас — поведінка обох старих копій: солвер пише
    діаметр саме туди, тому здогадка правильна для власних файлів і не
    гірша за помилку для чужих.
    """
    for column in frame.columns:
        low = str(column).lower()
        if any(hint in low for hint in DIAMETER_COLUMN_HINTS):
            return column
    return frame.columns[-1] if len(frame.columns) else None


def is_index(value, table):
    """Чи є значення індексом у таблиці типорозмірів, а не діаметром."""
    if not table or pd.isnull(value):
        return False
    try:
        number = float(value)
    except (ValueError, TypeError):
        return False
    return number.is_integer() and 0 <= int(number) < len(table)


def real_diameter(value, table):
    """Значення зі стовпця рішення, переведене в діаметр за таблицею."""
    return float(table[int(float(value))]) if is_index(value, table) else value


def with_real_diameters(frame, table, column=None):
    """Копія таблиці, у якій індекси типорозмірів замінені діаметрами."""
    column = column or diameter_column(frame)
    if column is None or not table:
        return frame.copy()

    result = frame.copy()
    result[column] = result[column].apply(lambda v: real_diameter(v, table))
    return result


def display_value(value):
    """Число — без зайвих нулів, два знаки після коми; решта — як є."""
    if pd.isnull(value):
        return ""
    try:
        number = float(value)
    except (ValueError, TypeError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


def display_name(value):
    """Назва вузла чи труби: `12.0` з CSV показується як `12`."""
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def for_display(frame):
    """Копія таблиці, готова до показу: назви як назви, числа як числа."""
    result = frame.copy()
    for column in result.columns:
        low = str(column).lower()
        if any(hint in low for hint in NAME_COLUMN_HINTS):
            result[column] = result[column].apply(display_name)
        else:
            result[column] = result[column].apply(display_value)
    return result
