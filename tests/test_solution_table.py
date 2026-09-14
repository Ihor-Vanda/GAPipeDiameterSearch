"""Розбір `solution.csv` — одне правило на карту й на таблицю.

Файл рішення читали двоє: `netdraw.diameters_from_table` і перегляд
результатів. Обидва шукали стовпець діаметра власним циклом по тому
самому списку підказок і обидва самі вирішували, індекс у клітинці чи
діаметр. Друга копія жила в модулі з tkinter, тому не перевірялась (§27.5).

Найтонше місце — саме це «індекс чи діаметр». Значення 4 у файлі з
таблицею Ханою — це п'ятий типорозмір (30 дюймів), а не діаметр 4; а
значення 12 — уже діаметр, бо індексу 12 в таблиці з шести рядків немає.
Правило неоднозначне за побудовою, тому важливо, щоб воно було одне.
"""

import pandas as pd
import pytest

import netdraw
from solution_table import (
    diameter_column, display_name, display_value, for_display, is_index,
    real_diameter, with_real_diameters,
)

HANOI = [12.0, 16.0, 20.0, 24.0, 30.0, 40.0]


def frame(**columns):
    return pd.DataFrame(columns)


# --- пошук стовпця ----------------------------------------------------

@pytest.mark.parametrize("name", ["Diameter", "diam", "DIAM_mm", "Діаметр",
                                  "діаметр_мм", "d_opt"])
def test_a_hinted_column_is_found(name):
    assert diameter_column(frame(Pipe=["P1"], **{name: [300]})) == name


def test_without_a_hint_the_last_column_is_used():
    """Так робили обидві старі копії: солвер пише діаметр останнім."""
    assert diameter_column(frame(Pipe=["P1"], Length=[100], X=[300])) == "X"


def test_the_first_hinted_column_wins():
    table = frame(Pipe=["P1"], d_old=[300], diam_new=[400])
    assert diameter_column(table) == "d_old"


def test_an_empty_frame_has_no_diameter_column():
    assert diameter_column(pd.DataFrame()) is None


# --- індекс чи діаметр ------------------------------------------------

@pytest.mark.parametrize("value", [0, 3, 5, 5.0, "4"])
def test_an_integer_inside_the_table_is_an_index(value):
    assert is_index(value, HANOI) is True


@pytest.mark.parametrize("value", [-1, 6, 100, 12.5, 300.0])
def test_anything_outside_the_table_is_a_diameter(value):
    assert is_index(value, HANOI) is False


def test_without_a_table_nothing_is_an_index():
    """Інтерфейс може не мати таблиці цін — тоді значення беруться як є."""
    assert is_index(3, None) is False
    assert is_index(3, []) is False
    assert real_diameter(3, None) == 3


def test_missing_and_non_numeric_values_are_left_alone():
    assert is_index(None, HANOI) is False
    assert is_index(float("nan"), HANOI) is False
    assert is_index("DN300", HANOI) is False
    assert real_diameter("DN300", HANOI) == "DN300"


def test_an_index_becomes_the_real_diameter():
    assert real_diameter(4, HANOI) == 30.0
    assert real_diameter(0, HANOI) == 12.0


def test_the_ambiguity_resolves_the_same_way_both_times():
    """12 — це діаметр (індексу 12 немає), 4 — це індекс. Саме цю пару
    двох різних трактувань одного стовпця і треба тримати в одному місці."""
    assert real_diameter(12, HANOI) == 12
    assert real_diameter(4, HANOI) == 30.0


# --- переклад цілої таблиці -------------------------------------------

def test_indices_in_the_diameter_column_are_translated():
    table = frame(Pipe=["P1", "P2", "P3"], Diameter=[0, 4, 40])
    result = with_real_diameters(table, HANOI)
    assert result["Diameter"].tolist() == [12.0, 30.0, 40]


def test_other_columns_are_untouched():
    table = frame(Pipe=["P1"], Length=[4], Diameter=[4])
    result = with_real_diameters(table, HANOI)
    assert result["Length"].tolist() == [4]
    assert result["Diameter"].tolist() == [30.0]


def test_the_original_frame_is_not_modified():
    table = frame(Pipe=["P1"], Diameter=[4])
    with_real_diameters(table, HANOI)
    assert table["Diameter"].tolist() == [4]


def test_without_a_table_the_frame_comes_back_as_it_was():
    table = frame(Pipe=["P1"], Diameter=[4])
    assert with_real_diameters(table, None)["Diameter"].tolist() == [4]


def test_an_explicit_column_overrides_the_search():
    table = frame(Pipe=["P1"], d_old=[4], d_new=[5])
    result = with_real_diameters(table, HANOI, column="d_new")
    assert result["d_old"].tolist() == [4]
    assert result["d_new"].tolist() == [40.0]


# --- підготовка до показу ---------------------------------------------

@pytest.mark.parametrize("value,shown", [
    (12.0, "12"), (12.5, "12.50"), (12.345, "12.35"), (0, "0"),
    (None, ""), (float("nan"), ""), ("текст", "текст"),
])
def test_numbers_lose_their_empty_decimals(value, shown):
    assert display_value(value) == shown


@pytest.mark.parametrize("value,shown", [
    (12.0, "12"), ("12.0", "12"), ("J12", "J12"), (7, "7"),
])
def test_names_read_from_csv_do_not_gain_a_decimal_point(value, shown):
    """`pandas` читає стовпець вузлів як float, і `J12` перетворювався б
    на `12.0` — тому назви форматуються окремо від чисел."""
    assert display_name(value) == shown


def test_name_columns_are_detected_by_their_header():
    table = frame(**{"Node ID": [12.0], "Вузол": [3.0], "Pressure": [31.456]})
    result = for_display(table)
    assert result["Node ID"].tolist() == ["12"]
    assert result["Вузол"].tolist() == ["3"]
    assert result["Pressure"].tolist() == ["31.46"]


def test_display_does_not_modify_the_original():
    table = frame(Pressure=[31.456])
    for_display(table)
    assert table["Pressure"].tolist() == [31.456]


def test_everything_shown_is_a_string():
    table = frame(**{"ID": [1.0], "Diameter": [300], "Pressure": [None]})
    result = for_display(table)
    assert all(isinstance(v, str) for row in result.values for v in row)


# --- та сама відповідь, що в карти ------------------------------------

class FakeGeometry:
    def __init__(self, names):
        self.edge_pipe_names = names


def test_the_map_and_the_table_agree_on_every_pipe():
    """Головне твердження: карта й таблиця більше не можуть розійтись —
    вони рахують діаметр одним кодом."""
    table = frame(Pipe=["P1", "P2", "P3"], Diameter=[0, 4, 350])
    geometry = FakeGeometry(["P1", "P2", "P3"])

    on_map = netdraw.diameters_from_table(geometry, table, HANOI)
    in_table = with_real_diameters(table, HANOI)["Diameter"].tolist()
    assert on_map == [float(v) for v in in_table]


def test_a_pipe_missing_from_the_solution_is_drawn_as_unknown():
    table = frame(Pipe=["P1"], Diameter=[4])
    geometry = FakeGeometry(["P1", "P2"])

    on_map = netdraw.diameters_from_table(geometry, table, HANOI)
    assert on_map == [30.0, netdraw.UNKNOWN_DIAMETER]


def test_the_map_follows_the_graph_order_not_the_file_order():
    """Порядок ребер графа й порядок рядків у файлі різні — саме це
    розв'язували чотири старі копії малювання, кожна по-своєму."""
    table = frame(Pipe=["P1", "P2"], Diameter=[0, 5])
    assert netdraw.diameters_from_table(FakeGeometry(["P2", "P1"]), table,
                                        HANOI) == [40.0, 12.0]
