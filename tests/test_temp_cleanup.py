"""Прибирання тимчасових каталогів воркерів.

`clean_temp_root` викликається чотири рази за прогін — на початку й у
`finally` обох точок входу, — і не мала жодного тесту. При цьому вона
робить дві небезпечні речі: змінює поточний каталог процесу і видаляє
дерево файлів. Помилка в такій функції дорога незалежно від того, чи
хтось її помітить.

Каталог визначається **відносно поточного каталогу**. Це і робить тести
можливими без підмін: `monkeypatch.chdir(tmp_path)` переносить увесь
корінь у тимчасове місце.

З етапу 28 корінь у кожного прогону **свій**:
`_temp_sim_data/run_<pid>`. Доти він був спільний, і два прогони з
однієї теки зносили файли одне одному — див. блок «два прогони поруч»
наприкінці файлу, який і є регресією на цей дефект.
"""

import os

import pytest

from pipeopt.solver.worker import (RUN_DIR_ENV, claim_temp_root,
                                   clean_temp_root, get_temp_root)


@pytest.fixture(autouse=True)
def no_inherited_run_dir(monkeypatch):
    """Кожен тест рахує корінь сам, а не успадковує чужий.

    Без цього перший тест, який викликає `claim_temp_root`, лишив би
    змінну середовища всім наступним — і вони б чистили не свій каталог.
    """
    monkeypatch.delenv(RUN_DIR_ENV, raising=False)


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """Корінь **цього прогону** всередині `tmp_path`, із двома воркерами."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "_temp_sim_data" / f"run_{os.getpid()}"

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
    assert get_temp_root() == str(
        tmp_path / "_temp_sim_data" / f"run_{os.getpid()}")


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


# --- два прогони поруч -------------------------------------------------

def test_each_run_gets_its_own_root(tmp_path, monkeypatch):
    """Корінь містить PID, інакше два прогони писали б в один каталог."""
    monkeypatch.chdir(tmp_path)

    assert f"run_{os.getpid()}" in get_temp_root()


def test_cleaning_one_run_does_not_touch_another(tmp_path, monkeypatch):
    """Регресія на дефект, через який один прогін убивав інший.

    Раніше `clean_temp_root` зносила **спільний** `_temp_sim_data`, тому
    прогін, що фінішував першим, забирав файли з-під того, хто ще
    працював; другий падав на передпольотній перевірці з `No such file or
    directory`, і це виглядало як помилка вхідних даних.
    """
    monkeypatch.chdir(tmp_path)
    shared = tmp_path / "_temp_sim_data"
    neighbour = shared / "run_999999"
    neighbour.mkdir(parents=True)
    (neighbour / "temp_rpt_1.rpt").write_text("сусід ще працює", encoding="utf-8")

    mine = claim_temp_root()
    (os.path.join(mine, "temp_rpt_2.rpt"))
    clean_temp_root()

    assert not os.path.exists(mine), "свій корінь мав зникнути"
    assert (neighbour / "temp_rpt_1.rpt").exists(), (
        "чужий прогін не має права постраждати від чужого прибирання")


def test_the_shared_root_survives_while_a_neighbour_is_there(tmp_path,
                                                             monkeypatch):
    monkeypatch.chdir(tmp_path)
    shared = tmp_path / "_temp_sim_data"
    (shared / "run_999999").mkdir(parents=True)

    claim_temp_root()
    clean_temp_root()

    assert shared.exists(), "спільний корінь ще комусь потрібен"


def test_the_shared_root_goes_away_when_it_empties(tmp_path, monkeypatch):
    """Інакше `_temp_sim_data` лишався б назавжди після кожного прогону."""
    monkeypatch.chdir(tmp_path)

    claim_temp_root()
    clean_temp_root()

    assert not (tmp_path / "_temp_sim_data").exists()


def test_workers_share_the_root_of_their_run(tmp_path, monkeypatch):
    """Воркер — окремий процес з іншим PID; корінь він бере зі змінної."""
    monkeypatch.chdir(tmp_path)
    mine = claim_temp_root()

    monkeypatch.setattr(os, "getpid", lambda: 4242)

    assert get_temp_root() == mine


def test_claiming_wipes_a_leftover_of_a_dead_run_with_the_same_pid(
        tmp_path, monkeypatch):
    """PID-и переживають процеси, і каталог міг лишитись від померлого."""
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "_temp_sim_data" / f"run_{os.getpid()}"
    stale.mkdir(parents=True)
    (stale / "temp_rpt_old.rpt").write_text("сміття", encoding="utf-8")

    claim_temp_root()

    assert not (stale / "temp_rpt_old.rpt").exists()


def test_claiming_twice_keeps_the_files_of_the_running_run(tmp_path,
                                                           monkeypatch):
    """`OptimizationRunner.prepare` закріплює корінь удруге.

    Якби другий виклик чистив каталог, він видалив би файли, якими
    головний процес уже користується.
    """
    monkeypatch.chdir(tmp_path)
    root = claim_temp_root()
    live = os.path.join(root, "temp_rpt_live.rpt")
    with open(live, "w", encoding="utf-8") as handle:
        handle.write("прогін іде")

    claim_temp_root()

    assert os.path.exists(live)
