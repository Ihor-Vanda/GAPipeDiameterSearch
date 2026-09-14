"""Прибирання тимчасових каталогів воркерів.

`clean_temp_root` викликається чотири рази за прогін — на початку й у
`finally` обох точок входу, — і не мала жодного тесту. При цьому вона
робить дві небезпечні речі: змінює поточний каталог процесу і видаляє
дерево файлів. Помилка в такій функції дорога незалежно від того, чи
хтось її помітить.

Каталог визначається як `os.path.abspath("_temp_sim_data")`, тобто
**відносно поточного каталогу**. Це і робить тести можливими без
підмін: `monkeypatch.chdir(tmp_path)` переносить увесь корінь у
тимчасове місце.
"""

import os

import pytest

from analytical_solver.worker import clean_temp_root, get_temp_root


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """Корінь тимчасових файлів усередині `tmp_path`, із двома воркерами."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "_temp_sim_data"

    for pid in (101, 202):
        worker = root / f"worker_{pid}"
        worker.mkdir(parents=True)
        (worker / f"temp_rpt_{pid}.rpt").write_text("звіт", encoding="utf-8")
        (worker / f"temp_bin_{pid}.bin").write_bytes(b"\x00\x01")
        (worker / f"Hanoi_worker_{pid}.inp").write_text("[JUNCTIONS]", encoding="utf-8")

    main = root / "main_process"
    main.mkdir()
    (main / "temp_rpt_1.rpt").write_text("звіт головного процесу", encoding="utf-8")

    return root


# --- де саме шукається корінь ------------------------------------------

def test_the_root_is_relative_to_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_temp_root() == str(tmp_path / "_temp_sim_data")


def test_the_root_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(get_temp_root())


# --- звичайне прибирання -----------------------------------------------

def test_the_whole_root_is_removed(temp_root):
    clean_temp_root()
    assert not temp_root.exists()


def test_nothing_outside_the_root_is_touched(temp_root, tmp_path):
    """Поруч із коренем лежать результати прогонів — їх видалення було б
    втратою даних, а не прибиранням."""
    assert temp_root.exists()
    keep = tmp_path / "OutputDataExperiments" / "run"
    keep.mkdir(parents=True)
    (keep / "solution_champion.csv").write_text("Pipe ID,Diameter", encoding="utf-8")
    (tmp_path / "config.py").write_text("# код", encoding="utf-8")

    clean_temp_root()

    assert (keep / "solution_champion.csv").exists()
    assert (tmp_path / "config.py").exists()


def test_a_missing_root_is_not_an_error(tmp_path, monkeypatch):
    """Перший виклик у прогоні йде до створення каталогу."""
    monkeypatch.chdir(tmp_path)
    clean_temp_root()
    assert not (tmp_path / "_temp_sim_data").exists()


def test_cleaning_twice_is_not_an_error(temp_root):
    clean_temp_root()
    clean_temp_root()
    assert not temp_root.exists()


def test_files_of_every_kind_epanet_leaves_are_removed(temp_root):
    """`.rpt`, `.bin`, `.inp`, `.out` — саме ці розширення й лишає Toolkit."""
    extra = temp_root / "worker_303"
    extra.mkdir()
    for name in ("temp.out", "temp_rpt_303.rpt", "temp_bin_303.bin", "net.inp"):
        (extra / name).write_text("x", encoding="utf-8")

    clean_temp_root()
    assert not temp_root.exists()


# --- поточний каталог --------------------------------------------------

def test_a_current_directory_inside_the_root_is_left_first(temp_root):
    """Воркер робить `os.chdir` у свій каталог, і на Windows каталог, у
    якому хтось стоїть, видалити не можна. Тому функція спершу виходить
    з кореня — інакше прибирання тихо не відбувалось би."""
    os.chdir(temp_root / "worker_101")

    clean_temp_root()

    assert not temp_root.exists()
    assert not os.path.abspath(os.getcwd()).startswith(str(temp_root))


def test_a_current_directory_outside_the_root_is_kept(temp_root):
    """І навпаки: якщо процес стоїть не в корені, каталог не мусить
    змінюватись — інакше відносні шляхи прогону поїхали б."""
    assert temp_root.exists()
    before = os.getcwd()
    clean_temp_root()
    assert os.getcwd() == before


# --- те, чого функція більше не робить ---------------------------------

def test_temp_files_in_the_current_directory_are_no_longer_hunted(tmp_path,
                                                                  monkeypatch):
    """Раніше функція сканувала **поточний** каталог на `temp*.rpt` тощо —
    це був наслідок §29.1 і §29.2, коли `plot.py` і `WaterSimulator`
    писали робочі файли в корінь проєкту. Після тих правок жоден код
    цього не робить, а файли під коренем прибирає обхід нижче.

    Тест фіксує саме те, що цикл прибрано: файл із такою назвою поза
    коренем — це файл користувача, і прибирання його не стосується.
    """
    monkeypatch.chdir(tmp_path)
    mine = tmp_path / "temp_rpt_mine.rpt"
    mine.write_text("мій файл", encoding="utf-8")

    clean_temp_root()

    assert mine.exists()


# --- невдача прибирання ------------------------------------------------

def test_a_failure_to_remove_the_root_is_reported(temp_root, monkeypatch, capsys):
    """`ignore_errors=True` ковтав невдачу, і корінь лишався непоміченим:
    саме так `_temp_sim_data/main_process` жив від прогону до прогону.
    Тепер прибирання каже, що не змогло (§31.6)."""
    import shutil

    def refuse(_path, ignore_errors=False):
        if not ignore_errors:
            raise OSError("зайнято іншим процесом")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    clean_temp_root()

    output = capsys.readouterr().out
    assert "Не вдалось видалити" in output
    assert str(temp_root) in output


def test_a_successful_cleanup_says_nothing(temp_root, capsys):
    """Успіх мусить бути тихим — інакше кожен прогін друкував би рядок
    ні про що."""
    assert temp_root.exists()
    clean_temp_root()
    assert capsys.readouterr().out == ""
