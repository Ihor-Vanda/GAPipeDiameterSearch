"""Адаптер до EPANET: одиниці, вартість, тиски, критичний вузол.

Модуль мав 16 % покриття — найнижче серед усього, що не є інтерфейсом, —
і водночас це єдине місце, де числа переходять між системами одиниць.
Саме тут жила помилка, що обійшлась дорожче за всі інші в цьому проєкті:
таблиця цін Ханою в дюймах, а прогін без `--units in` трактував її як
міліметри, тобто діаметри виходили в 25 разів меншими. Симуляція при
цьому не падала — вона повертала тиски близько −10⁸ м, і результат
експортувався як успішний (§18.5).

Головний тест тут — `test_the_same_network_in_us_units_gives_the_same
_pressures`: та сама мережа, записана в GPM і в CMH, мусить дати ті самі
тиски **в метрах**. Це перевіряє три множники переведення одночасно, і
жоден інший тест у проєкті цього не робить.

Тести використовують справжній EPANET, тому пропускаються, якщо його
немає.
"""

import os

import numpy as np
import pytest

wntr = pytest.importorskip("wntr", reason="EPANET/wntr недоступний")

from pipeopt.config import SolverConfig  # noqa: E402
from pipeopt.reporting.cost_table import load_cost_table  # noqa: E402
from pipeopt.hydraulics.water_sim import WaterSimulator  # noqa: E402

HANOI_INP = os.path.join("InputData", "Hanoi", "Hanoi.inp")
HANOI_COSTS = os.path.join("InputData", "Hanoi", "costs.csv")

#: Ханой: 34 труби, 6 типорозмірів (12…40 дюймів), обмеження 30 м.
PIPES = 34
MAX_IDX = 5
H_MIN = 30.0

pytestmark = pytest.mark.skipif(
    not (os.path.exists(HANOI_INP) and os.path.exists(HANOI_COSTS)),
    reason="немає InputData/Hanoi",
)


@pytest.fixture(scope="module")
def cfg():
    config = SolverConfig(cost_file=HANOI_COSTS, unit_system="in", h_min=H_MIN)
    load_cost_table(config, silent=True)
    return config


@pytest.fixture(scope="module")
def sim(cfg, tmp_path_factory):
    scratch = tmp_path_factory.mktemp("epanet")
    return WaterSimulator(HANOI_INP, cfg, temp_dir=str(scratch))


# --- що модуль прочитав із мережі --------------------------------------

def test_the_network_is_read_completely(sim):
    assert sim.n_variables == PIPES
    assert len(sim.component_names) == PIPES
    assert len(sim.lengths) == PIPES
    assert sim.sources == ["1"]


def test_lengths_come_from_the_inp_not_from_the_cost_table(sim):
    assert all(length > 0 for length in sim.lengths)
    assert sum(sim.lengths) == pytest.approx(39420.0)


def test_the_options_come_from_the_cost_table(sim, cfg):
    assert sim.n_options == len(cfg.diameters_m) == 6
    assert sim.diams_m == cfg.diameters_m
    assert sim.costs == [cfg.costs[d] for d in cfg.diameters_raw]


# --- одиниці ----------------------------------------------------------

def test_hanoi_is_read_as_metric(sim):
    """Тонкість, через яку і виникла помилка §18.5: сама мережа метрична
    (CMH), а таблиця цін до неї — у дюймах. Це різні одиниці в одній
    задачі, і кожна конвертується своїм шляхом."""
    assert sim.is_us_units is False
    assert sim.diam_epanet_mult == 1000.0
    assert sim.press_si_mult == 1.0
    assert sim.hl_si_mult == 1.0


def test_the_same_network_in_us_units_gives_the_same_pressures(cfg, tmp_path):
    """Та сама мережа, записана в GPM: EPANET віддає тиск у psi і втрати у
    футах, тому модуль множить їх на 0.7032496 і 0.3048. Якщо котрийсь із
    трьох множників помилковий, тиски розійдуться — а це єдиний тест, що
    його зловить."""
    model = wntr.network.WaterNetworkModel(HANOI_INP)
    model.options.hydraulic.inpfile_units = "GPM"
    us_inp = str(tmp_path / "hanoi_gpm.inp")
    wntr.network.write_inpfile(model, us_inp, units="GPM")

    us = WaterSimulator(us_inp, cfg, temp_dir=str(tmp_path))
    assert us.is_us_units is True
    assert us.diam_epanet_mult == pytest.approx(39.3701)

    metric_stats = WaterSimulator(HANOI_INP, cfg,
                                  temp_dir=str(tmp_path)).get_stats([MAX_IDX] * PIPES)
    us_stats = us.get_stats([MAX_IDX] * PIPES)

    assert us_stats.p_min == pytest.approx(metric_stats.p_min, abs=0.05)
    assert us_stats.p_max == pytest.approx(metric_stats.p_max, abs=0.05)
    assert us_stats.crit_node == metric_stats.crit_node


# --- вартість ---------------------------------------------------------

def test_cost_is_length_times_unit_price(sim, cfg):
    """Незалежний перерахунок: вартість не залежить від гідравліки, тому
    її можна порахувати без EPANET і порівняти."""
    solution = [MAX_IDX] * PIPES
    expected = sum(length * cfg.costs[cfg.diameters_raw[idx]]
                   for length, idx in zip(sim.lengths, solution))
    assert sim.get_stats(solution).cost == pytest.approx(expected)


def test_a_thinner_solution_costs_less(sim):
    assert sim.get_stats([0] * PIPES).cost < sim.get_stats([MAX_IDX] * PIPES).cost


def test_cost_grows_with_every_step_up(sim):
    costs = [sim.get_stats([i] * PIPES).cost for i in range(6)]
    assert costs == sorted(costs)


def test_out_of_range_indices_are_clamped_not_an_error(sim):
    """Кіки можуть вийти за межі таблиці, і симулятор мусить це стерпіти,
    а не впасти посеред мільйона симуляцій."""
    top = sim.get_stats([MAX_IDX] * PIPES)
    assert sim.get_stats([99] * PIPES).cost == pytest.approx(top.cost)
    assert sim.get_stats([-7] * PIPES).cost == pytest.approx(
        sim.get_stats([0] * PIPES).cost)


def test_numpy_integers_are_accepted(sim):
    """Рішення приходять і списком, і масивом `numpy` — з кіків і з пулу."""
    as_list = sim.get_stats([3] * PIPES)
    as_array = sim.get_stats(np.full(PIPES, 3, dtype=np.int32))
    assert as_array.cost == pytest.approx(as_list.cost)
    assert as_array.p_min == pytest.approx(as_list.p_min)


# --- тиски ------------------------------------------------------------

def test_the_widest_solution_satisfies_the_constraint(sim):
    stats = sim.get_stats([MAX_IDX] * PIPES)
    assert stats.simulation_ok
    assert stats.is_feasible(H_MIN)
    assert stats.p_min == pytest.approx(49.62, abs=0.05)


def test_the_narrowest_solution_violates_it_badly(sim):
    """Усі мінімальні діаметри дають тиск близько −17 600 м. Симуляція при
    цьому **не падає** — саме тому `simulation_ok` не можна плутати з
    допустимістю."""
    stats = sim.get_stats([0] * PIPES)
    assert stats.simulation_ok
    assert not stats.is_feasible(H_MIN)
    assert stats.p_min < -1000


def test_the_critical_node_is_the_one_with_the_lowest_pressure(sim):
    stats = sim.get_stats([MAX_IDX] * PIPES)
    assert stats.crit_node in sim.c_idx_to_node_name.values()
    assert stats.p_min <= stats.p_max


def test_surplus_is_the_distance_to_the_constraint(sim):
    stats = sim.get_stats([MAX_IDX] * PIPES)
    assert stats.surplus(H_MIN) == pytest.approx(stats.p_min - H_MIN)


def test_pressure_rises_as_pipes_widen(sim):
    """Монотонність за діаметром — властивість, на якій тримається весь
    спуск: якщо її нема, «звузити трубу» перестає означати «втратити
    тиск»."""
    pressures = [sim.get_stats([i] * PIPES).p_min for i in range(6)]
    assert pressures == sorted(pressures)


# --- евристики й стан ---------------------------------------------------

def test_heuristics_are_losses_per_metre(sim):
    losses = sim.get_heuristics([MAX_IDX] * PIPES)
    assert len(losses) == PIPES
    assert all(loss >= 0 for loss in losses)
    assert max(losses) > 0


def test_thin_pipes_lose_more_per_metre_than_thick_ones(sim):
    """Хазен-Вільямс: втрати ~ 1/d^4.87, отже перехід на найменший
    типорозмір мусить підняти питомі втрати на кожній трубі."""
    thick = sim.get_heuristics([MAX_IDX] * PIPES)
    thin = sim.get_heuristics([0] * PIPES)
    assert sum(thin) > sum(thick)


def test_hydraulic_state_returns_a_flow_per_pipe(sim, cfg):
    real = [cfg.diameters_m[MAX_IDX]] * PIPES
    flows, feasible = sim.get_hydraulic_state(real)
    assert len(flows) == PIPES
    assert feasible is True
    assert any(abs(flow) > 0 for flow in flows)


def test_hydraulic_state_takes_metres_not_indices(sim, cfg):
    """Єдиний метод, який приймає **реальні діаметри**, а не індекси. Якби
    йому передали індекси, 0…5 метрів дали б зовсім іншу мережу — і
    помилки не було б видно."""
    by_metres, _ = sim.get_hydraulic_state([cfg.diameters_m[MAX_IDX]] * PIPES)
    by_index, _ = sim.get_hydraulic_state([float(MAX_IDX)] * PIPES)
    assert by_metres != by_index


# --- робочі файли ------------------------------------------------------

def test_scratch_files_stay_in_the_temp_directory(cfg, tmp_path, monkeypatch):
    """Регресія на §29.2: з `temp_dir=None` робочі файли EPANET писались у
    поточний каталог, тобто в корінь проєкту."""
    work = tmp_path / "cwd"
    work.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(work)

    WaterSimulator(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), HANOI_INP), cfg, temp_dir=str(scratch))

    assert list(work.iterdir()) == []
    assert any(name.startswith("temp_rpt_") for name in os.listdir(scratch))


def test_the_temp_directory_is_required(cfg):
    with pytest.raises(TypeError):
        WaterSimulator(HANOI_INP, cfg)


# --- закриття Toolkit ---------------------------------------------------

def test_closing_releases_the_toolkit(cfg, tmp_path):
    """Без явного закриття на Windows `_temp_sim_data` не видаляється:
    Toolkit тримає .rpt відкритим, поки живий симулятор (§31.1)."""
    simulator = WaterSimulator(HANOI_INP, cfg, temp_dir=str(tmp_path))
    assert simulator.api is not None

    simulator.close()
    assert simulator.api is None


def test_closing_twice_is_not_an_error(cfg, tmp_path):
    """`close()` викликається і з `OptimizationRunner.close`, і з
    `__del__`, тому мусить бути ідемпотентним."""
    simulator = WaterSimulator(HANOI_INP, cfg, temp_dir=str(tmp_path))
    simulator.close()
    simulator.close()
    assert simulator.api is None


def test_the_scratch_files_can_be_deleted_after_closing(cfg, tmp_path):
    """Саме те, чого не міг зробити `clean_temp_root` у CLI: видалити
    робочі файли симулятора головного процесу."""
    simulator = WaterSimulator(HANOI_INP, cfg, temp_dir=str(tmp_path))
    simulator.get_stats([MAX_IDX] * PIPES)
    scratch = [name for name in os.listdir(tmp_path) if name.startswith("temp_")]
    assert scratch

    simulator.close()
    for name in scratch:
        os.remove(tmp_path / name)
    assert not [n for n in os.listdir(tmp_path) if n.startswith("temp_")]
