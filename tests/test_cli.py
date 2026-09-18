"""Розбір аргументів CLI: єдина точка, де користувач задає прогін.

`main.py` мав **нуль відсотків** покриття. Причина структурна: розбір
аргументів жив усередині `main()` разом із запуском, і перевірити його
можна було лише провівши прогін цілком. На етапі 18 з нього винесені три
функції — `build_parser`, `apply_config_file`, `request_from_args`, — і
все, що перевіряється нижче, стало доступним за мілісекунди.

Чому це варто перевіряти. Прапорці тут задають не дрібниці, а режим
роботи:

* `--seed` разом із `--sequential` — єдиний спосіб отримати відтворюваний
  прогін (§12.4). Помилка в їх передачі не падає — вона тихо робить
  прогін невідтворюваним, і це виявляється лише тоді, коли не сходиться
  еталон;
* `--units` — саме його відсутність дала прогін із тисками −10⁸ (§18.5);
* `--config` накладається на аргументи, і правило «явний прапорець
  сильніший за файл» ніде, крім коду, не записане.
"""

import json

import pytest

from main import apply_config_file, build_parser, request_from_args


def parse(*argv):
    args, _ = build_parser().parse_known_args(list(argv))
    return args


# --- замовчування ------------------------------------------------------

def test_the_defaults_describe_a_runnable_hanoi_run():
    args = parse()

    assert args.inp.endswith("Hanoi.inp")
    assert args.costs.endswith("costs.csv")
    assert args.runs == 1
    assert args.hmin == 30.0
    assert args.run_mode == "analytical"


def test_a_run_is_not_reproducible_unless_asked():
    """Без `--seed` прогін навмисне випадковий — це задокументована поведінка."""
    args = parse()

    assert args.seed is None
    assert args.sequential is False


def test_the_units_flag_only_accepts_the_two_real_systems():
    """`--units` — той прапорець, забуття якого коштувало прогону з тисками −10⁸."""
    assert parse("--units", "in").units == "in"
    assert parse("--units", "mm").units == "mm"

    with pytest.raises(SystemExit):
        parse("--units", "inches")


def test_the_run_mode_is_restricted_to_the_two_implemented_ones():
    with pytest.raises(SystemExit):
        parse("--run_mode", "genetic")


def test_unknown_flags_do_not_kill_the_run():
    """`parse_known_args`: зайвий прапорець не має зупиняти багатогодинний прогін."""
    args = parse("--seed", "42", "--what_is_this", "7")

    assert args.seed == 42


# --- конфіг із файлу ---------------------------------------------------

@pytest.fixture
def config_file(tmp_path):
    def write(payload):
        path = tmp_path / "run.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)
    return write


def test_a_config_file_fills_in_what_the_command_line_omitted(config_file):
    args = parse("--config", config_file({"max_sims": 15_000_000, "hmin": 25.0}))

    apply_config_file(args, argv=["main.py", "--config", "x.json"])

    assert args.max_sims == 15_000_000
    assert args.hmin == 25.0


def test_an_explicit_flag_beats_the_config_file(config_file):
    """Інакше конфіг серії мовчки затирав би `--seed`, яким її й розрізняють."""
    args = parse("--config", config_file({"seed": 1}), "--seed", "42")

    apply_config_file(args, argv=["main.py", "--config", "x.json", "--seed", "42"])

    assert args.seed == 42


def test_the_equals_form_of_a_flag_counts_as_explicit(config_file):
    args = parse("--config", config_file({"seed": 1}), "--seed=42")

    apply_config_file(args, argv=["main.py", "--seed=42"])

    assert args.seed == 42, "`--seed=42` — це те саме, що `--seed 42`"


def test_an_unknown_key_in_the_config_is_skipped(config_file):
    """Конфіги пишуться руками; падати через зайвий рядок гірше, ніж його проковтнути."""
    args = parse("--config", config_file({"hmin": 25.0, "colour": "blue"}))

    apply_config_file(args, argv=["main.py"])

    assert args.hmin == 25.0
    assert not hasattr(args, "colour")


def test_a_missing_config_file_is_not_an_error():
    args = parse("--config", "/no/such/file.json")

    apply_config_file(args, argv=["main.py"])

    assert args.hmin == 30.0


def test_no_config_at_all_changes_nothing():
    args = parse("--hmin", "42")

    apply_config_file(args, argv=["main.py", "--hmin", "42"])

    assert args.hmin == 42.0


# --- запит на прогін ---------------------------------------------------

def test_the_request_carries_the_flags_through():
    args = parse("--hmin", "25", "--units", "in", "--runs", "3", "--v_opt", "1.5")

    request = request_from_args(args)

    assert (request.h_min, request.units, request.runs, request.v_opt) == \
        (25.0, "in", 3, 1.5)


def test_paths_become_absolute():
    """Воркери стартують у власних тимчасових каталогах — відносний шлях там мертвий."""
    import os

    request = request_from_args(parse("--inp", "InputData/Hanoi/Hanoi.inp"))

    assert os.path.isabs(request.inp_file)
    assert os.path.isabs(request.cost_file)


def test_sequential_forces_a_single_island():
    """Відтворюваність можлива лише в одному процесі — `--cores` тут не діє."""
    request = request_from_args(parse("--sequential", "--cores", "8"))

    assert request.effective_cores == 1


def test_sequential_without_a_time_limit_stops_only_on_simulations():
    """Годинник робить точку зупинки залежною від завантаження машини (§12.4)."""
    request = request_from_args(parse("--sequential"))

    assert request.solver_time_limit == float("inf")


def test_a_zero_time_limit_means_unlimited():
    assert request_from_args(parse("--time_limit", "0")).solver_time_limit == float("inf")


def test_an_explicit_time_limit_survives():
    assert request_from_args(parse("--time_limit", "90")).solver_time_limit == 90.0


def test_a_parallel_run_without_a_limit_derives_one_later():
    """`None` означає «вивести з калібрування», а не «без обмеження»."""
    assert request_from_args(parse("--cores", "4")).solver_time_limit is None


def test_zero_cores_means_every_core():
    import multiprocessing

    request = request_from_args(parse("--cores", "0"))

    assert request.effective_cores == multiprocessing.cpu_count()


def test_an_impossible_request_is_refused_at_construction():
    """`RunRequest` перевіряє себе сам — помилка видна до першої симуляції."""
    with pytest.raises(ValueError):
        request_from_args(parse("--runs", "0"))
