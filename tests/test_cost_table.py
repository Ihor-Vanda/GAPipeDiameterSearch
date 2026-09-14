"""Розбір таблиці цін — один парсер на застосунок і на інтерфейс.

Розбір жив усередині `load_cost_table`, який на будь-якій помилці робить
`sys.exit(1)`. Інтерфейсу така реакція не годиться: жива карта мережі
малюється до створення конфігу, тому GUI мав власний читач
(`_get_actual_diams`) — слабший: він не знав ні `;`, ні варіантів назв
стовпців, ні пробілів у заголовках, тобто на тих самих файлах давав
інший результат (§27.3).

Тепер розбір один (`read_cost_table`), а різниця лишилась тільки в
реакції на помилку: `load_cost_table` завершує процес, `diameter_table`
повертає `None`. Саме цю пару властивостей і фіксують тести.
"""

import os

import pandas as pd
import pytest

from config import SolverConfig
from cost_table import diameter_table, load_cost_table, read_cost_table

HANOI_COSTS = os.path.join("InputData", "Hanoi", "costs.csv")


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- розділювачі ------------------------------------------------------

def test_comma_separated_table(tmp_path):
    path = write(tmp_path / "c.csv", "Diameter,Cost\n300,10\n400,20\n")
    frame, diam, cost = read_cost_table(path)
    assert (diam, cost) == ("Diameter", "Cost")
    assert frame[diam].tolist() == [300, 400]


def test_semicolon_separated_table(tmp_path):
    """Старий читач інтерфейсу бачив тут один стовпець і вертав `None`."""
    path = write(tmp_path / "c.csv", "Diameter;Cost\n300;10\n400;20\n")
    frame, diam, cost = read_cost_table(path)
    assert (diam, cost) == ("Diameter", "Cost")
    assert frame[diam].tolist() == [300, 400]
    assert diameter_table(path) == [300.0, 400.0]


def test_a_tab_separated_table_falls_back_to_sniffing(tmp_path):
    """Третя спроба — автовизначення — була недосяжна: `read_csv` на такому
    файлі не падає, а повертає один стовпець, тому `except` не виконувався
    (§27.4).

    Ширше за табуляцію автовизначення `pandas` не тягне: на `Diameter|Cost`
    воно вибирає `t`. Тому обіцяне тут — кома, `;` і табуляція, не «будь-який
    розділювач».
    """
    path = write(tmp_path / "c.csv", "Diameter\tCost\n300\t10\n")
    frame, diam, cost = read_cost_table(path)
    assert frame[diam].tolist() == [300]
    assert frame[cost].tolist() == [10]


def test_a_file_that_cannot_be_read_at_all_is_reported(tmp_path):
    with pytest.raises((ValueError, OSError)):
        read_cost_table(str(tmp_path))


# --- назви стовпців ---------------------------------------------------

@pytest.mark.parametrize("header", ["Diameter,Cost", "Diam,cost", "D,Price",
                                    "diameter,price", "size,UnitCost"])
def test_every_documented_column_name_is_accepted(tmp_path, header):
    path = write(tmp_path / "c.csv", f"{header}\n300,10\n")
    frame, diam, cost = read_cost_table(path)
    assert frame[diam].tolist() == [300]
    assert frame[cost].tolist() == [10]


@pytest.mark.parametrize("header", ["Diameter (inch),Unit Cost",
                                    "DIAM_mm,COST_usd",
                                    "Діаметр,Ціна"])
def test_a_column_name_with_extra_words_is_found_by_substring(tmp_path, header):
    """Точного збігу тут немає — саме цього не бачив `load_cost_table`, хоча
    читач інтерфейсу бачив. Тепер приймається обидва способи (§27.3)."""
    path = write(tmp_path / "c.csv", f"{header}\n300,10\n")
    frame, diam, cost = read_cost_table(path)
    assert frame[diam].tolist() == [300]
    assert frame[cost].tolist() == [10]


def test_an_exact_name_wins_over_a_substring(tmp_path):
    """Порядок зафіксований, бо від нього залежить, який стовпець стане
    діаметром у файлі, де підходять обидва."""
    path = write(tmp_path / "c.csv", "Diameter_old,Diameter,Cost\n1,300,10\n")
    _, diam, _ = read_cost_table(path)
    assert diam == "Diameter"


def test_whitespace_in_headers_is_stripped(tmp_path):
    path = write(tmp_path / "c.csv", " Diameter , Cost \n300,10\n")
    _, diam, cost = read_cost_table(path)
    assert (diam, cost) == ("Diameter", "Cost")


def test_a_table_without_a_diameter_column_is_an_error(tmp_path):
    path = write(tmp_path / "c.csv", "Name,Cost\nP1,10\n")
    with pytest.raises(KeyError):
        read_cost_table(path)


def test_a_table_without_a_cost_column_is_an_error(tmp_path):
    path = write(tmp_path / "c.csv", "Diameter,Length\n300,10\n")
    with pytest.raises(KeyError):
        read_cost_table(path)


# --- порядок ----------------------------------------------------------

def test_rows_come_back_sorted_by_diameter(tmp_path):
    """Індекс у рішенні — це позиція в цьому списку, тому порядок є частиною
    контракту: файл із переставленими рядками мусить дати той самий список."""
    path = write(tmp_path / "c.csv", "Diameter,Cost\n400,20\n300,10\n600,40\n")
    frame, diam, _ = read_cost_table(path)
    assert frame[diam].tolist() == [300, 400, 600]
    assert diameter_table(path) == [300.0, 400.0, 600.0]


# --- м'яка обгортка для інтерфейсу ------------------------------------

def test_a_missing_file_is_not_an_error_for_the_interface(tmp_path):
    assert diameter_table(str(tmp_path / "nope.csv")) is None


def test_no_file_at_all_is_not_an_error_for_the_interface():
    assert diameter_table(None) is None
    assert diameter_table("") is None


def test_a_broken_table_leaves_the_map_without_real_diameters(tmp_path):
    path = write(tmp_path / "c.csv", "Name,Length\nP1,10\n")
    assert diameter_table(path) is None


def test_non_numeric_diameters_are_not_an_error_for_the_interface(tmp_path):
    path = write(tmp_path / "c.csv", "Diameter,Cost\nDN300,10\n")
    assert diameter_table(path) is None


# --- те саме, але для застосунку: помилка спиняє прогін ---------------

def test_load_cost_table_exits_when_the_file_is_missing(tmp_path):
    config = SolverConfig(cost_file=str(tmp_path / "nope.csv"))
    with pytest.raises(SystemExit):
        load_cost_table(config, silent=True)


def test_load_cost_table_exits_when_the_columns_are_wrong(tmp_path):
    path = write(tmp_path / "c.csv", "Name,Length\nP1,10\n")
    config = SolverConfig(cost_file=path)
    with pytest.raises(SystemExit):
        load_cost_table(config, silent=True)


# --- один парсер на двох ----------------------------------------------

def test_the_interface_and_the_solver_see_the_same_diameters(tmp_path):
    """Головне твердження етапу: два шляхи дають один список.

    До зведення до одного парсера це не було так на жодному файлі з `;`
    або з назвою стовпця, відмінною від `Diameter`.
    """
    path = write(tmp_path / "c.csv", "Diam;Price\n400;20\n300;10\n")
    config = SolverConfig(cost_file=path, unit_system="mm")
    load_cost_table(config, silent=True)
    assert diameter_table(path) == config.diameters_raw


@pytest.mark.skipif(not os.path.exists(HANOI_COSTS), reason="немає InputData")
def test_the_two_paths_agree_on_the_real_hanoi_table():
    config = SolverConfig(cost_file=HANOI_COSTS, unit_system="in")
    load_cost_table(config, silent=True)
    assert diameter_table(HANOI_COSTS) == config.diameters_raw
    assert config.diameters_raw == [12.0, 16.0, 20.0, 24.0, 30.0, 40.0]


# --- те, що load_cost_table робить понад розбір -----------------------

def test_millimetres_are_converted_to_metres(tmp_path):
    path = write(tmp_path / "c.csv", "Diameter,Cost\n300,10\n")
    config = SolverConfig(cost_file=path, unit_system="mm")
    load_cost_table(config, silent=True)
    assert config.diameters_m == [0.3]


def test_inches_are_converted_to_metres(tmp_path):
    path = write(tmp_path / "c.csv", "Diameter,Cost\n12,10\n")
    config = SolverConfig(cost_file=path, unit_system="in")
    load_cost_table(config, silent=True)
    assert config.diameters_m == pytest.approx([0.3048])


def test_costs_are_keyed_by_raw_diameter(tmp_path):
    path = write(tmp_path / "c.csv", "Diameter,Cost\n400,20\n300,10\n")
    config = SolverConfig(cost_file=path, unit_system="mm")
    load_cost_table(config, silent=True)
    assert config.costs == {300: 10, 400: 20}


def test_the_diameter_table_keeps_the_index_contract(tmp_path):
    """`diameters_raw[i]`, `diameters_m[i]` і `diameter_table(...)[i]` — це
    одна й та сама труба: на цьому тримається переклад індексів у рішенні."""
    path = write(tmp_path / "c.csv", "Diameter,Cost\n600,40\n300,10\n400,20\n")
    config = SolverConfig(cost_file=path, unit_system="mm")
    load_cost_table(config, silent=True)

    table = diameter_table(path)
    assert table == config.diameters_raw
    assert config.diameters_m == pytest.approx([d / 1000.0 for d in table])
    assert sorted(table) == table


def test_the_parser_does_not_mutate_the_file(tmp_path):
    path = write(tmp_path / "c.csv", "Diameter,Cost\n400,20\n300,10\n")
    before = pd.read_csv(path).to_dict()
    read_cost_table(path)
    diameter_table(path)
    assert pd.read_csv(path).to_dict() == before
