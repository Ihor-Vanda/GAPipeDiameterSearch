"""Спільний сценарій запуску: контракт запиту й результату.

`RunRequest.from_text_fields` — це та сама толерантна логіка розбору, що
жила всередині `gui.run_optimization` (`try: int(...) except: 0`). Вона
винесена саме для того, щоб її можна було перевірити без tkinter, якого в
CI немає.
"""

import multiprocessing

import pytest

from runner import OptimizationRunner, RunRequest, RunResult


def make_request(**kw):
    kw.setdefault("inp_file", "net.inp")
    kw.setdefault("cost_file", "costs.csv")
    return RunRequest(**kw)


# --- валідація --------------------------------------------------------

def test_defaults_are_the_documented_ones():
    r = make_request()
    assert (r.h_min, r.units, r.v_opt, r.runs, r.cores) == (30.0, "mm", 1.0, 1, 0)
    assert r.max_sims is None
    assert r.run_mode == "analytical"
    assert r.sequential is False


def test_unknown_run_mode_is_rejected():
    with pytest.raises(ValueError, match="run_mode"):
        make_request(run_mode="genetic")


def test_zero_runs_is_rejected():
    with pytest.raises(ValueError, match="runs"):
        make_request(runs=0)


# --- розбір полів GUI -------------------------------------------------

def text_fields(**over):
    fields = dict(inp_file="net.inp", cost_file="costs.csv", hmin="30.0", units="mm",
                  v_opt="1.0", runs="3", cores="5", max_sims="40000",
                  run_mode="analytical")
    fields.update(over)
    return fields


def test_parses_normal_values():
    r = RunRequest.from_text_fields(**text_fields())
    assert (r.h_min, r.v_opt, r.runs, r.cores, r.max_sims) == (30.0, 1.0, 3, 5, 40000)


@pytest.mark.parametrize("bad", ["", "   ", "abc", None])
def test_broken_max_sims_falls_back_to_budget_by_network_class(bad):
    """Порожнє поле означає «бюджет за класом мережі», тобто None."""
    assert RunRequest.from_text_fields(**text_fields(max_sims=bad)).max_sims is None


def test_zero_max_sims_also_means_default_budget():
    assert RunRequest.from_text_fields(**text_fields(max_sims="0")).max_sims is None


@pytest.mark.parametrize("bad", ["", "abc", None])
def test_broken_cores_means_auto(bad):
    assert RunRequest.from_text_fields(**text_fields(cores=bad)).cores == 0


@pytest.mark.parametrize("bad", ["", "abc", "0", "-4"])
def test_broken_runs_falls_back_to_one(bad):
    assert RunRequest.from_text_fields(**text_fields(runs=bad)).runs == 1


def test_broken_hmin_falls_back_to_default():
    assert RunRequest.from_text_fields(**text_fields(hmin="")).h_min == 30.0


def test_whitespace_is_tolerated():
    r = RunRequest.from_text_fields(**text_fields(hmin=" 25.5 ", runs=" 2 "))
    assert r.h_min == 25.5
    assert r.runs == 2


# --- похідні величини -------------------------------------------------

def test_zero_cores_means_all_available():
    assert make_request(cores=0).effective_cores == multiprocessing.cpu_count()


def test_explicit_cores_are_respected():
    assert make_request(cores=3).effective_cores == 3


def test_sequential_forces_one_core():
    assert make_request(cores=8, sequential=True).effective_cores == 1


def test_sequential_drops_the_wall_clock_limit():
    """Часовий бюджет за замовчуванням виводиться з калібрування, тому
    робить точку зупинки залежною від завантаження машини (§12.4)."""
    assert make_request(sequential=True).solver_time_limit == float("inf")


def test_explicit_time_limit_survives_sequential():
    assert make_request(sequential=True, time_limit=120).solver_time_limit == 120


def test_non_positive_time_limit_means_unlimited():
    assert make_request(time_limit=0).solver_time_limit == float("inf")
    assert make_request(time_limit=-1).solver_time_limit == float("inf")


def test_unset_time_limit_stays_unset_in_parallel_mode():
    assert make_request().solver_time_limit is None


# --- результат --------------------------------------------------------

def make_result(**kw):
    kw.setdefault("run_id", 1)
    kw.setdefault("cost", 6.0e6)
    kw.setdefault("pressure", 30.5)
    kw.setdefault("feasible", True)
    kw.setdefault("duration", 12.5)
    kw.setdefault("individual", [1, 2, 3])
    return RunResult(**kw)


def test_total_evals_reads_the_last_history_point():
    r = make_result(history=[{"evals": 10, "min_cost": 9}, {"evals": 99, "min_cost": 6}])
    assert r.total_evals == 99


def test_total_evals_is_zero_without_history():
    assert make_result().total_evals == 0


def test_status_reflects_feasibility():
    assert make_result(feasible=True).status == "OK"
    assert make_result(feasible=False).status == "FAIL"


def test_summary_row_has_the_columns_of_runs_summary_csv():
    assert set(make_result().as_summary_row()) == {"Run", "Cost", "Pressure", "Feasible", "Time"}


# --- вибір чемпіона ---------------------------------------------------

def test_champion_prefers_feasible_over_cheap():
    cheap_but_invalid = make_result(run_id=1, cost=1.0e6, feasible=False)
    pricier_but_valid = make_result(run_id=2, cost=9.0e6, feasible=True)

    best = OptimizationRunner.champion([cheap_but_invalid, pricier_but_valid])
    assert best.run_id == 2


def test_champion_takes_the_cheapest_among_feasible():
    results = [make_result(run_id=1, cost=7.0e6), make_result(run_id=2, cost=6.0e6),
               make_result(run_id=3, cost=8.0e6)]
    assert OptimizationRunner.champion(results).run_id == 2


def test_champion_falls_back_to_infeasible_if_nothing_else():
    results = [make_result(run_id=1, cost=7.0e6, feasible=False),
               make_result(run_id=2, cost=6.0e6, feasible=False)]
    assert OptimizationRunner.champion(results).run_id == 2


def test_champion_of_nothing_is_none():
    assert OptimizationRunner.champion([]) is None


# --- зворотне відображення діаметрів ----------------------------------

def test_diameters_map_back_to_indices(sim):
    runner = OptimizationRunner(make_request())
    runner.config = sim.config

    table = sim.config.diameters_m
    assert runner._indices_of([table[2], table[0], table[3]]) == [2, 0, 3]


def test_unknown_diameter_falls_back_to_the_largest(sim):
    """Так поводився обидва оригінальні цикли: ValueError -> останній індекс."""
    runner = OptimizationRunner(make_request())
    runner.config = sim.config

    assert runner._indices_of([0.12345]) == [len(sim.config.diameters_m) - 1]


# --- контракт із солвером ---------------------------------------------
#
# Етап 5 переніс сценарій запуску, але не мусив змінити ані одного
# аргументу, з яким створюється `AnalyticalSolver`. Прогін на Hanoi це
# перевіряє лише опосередковано (і стохастично), тому аргументи
# зафіксовані тестом.

class _SpySolver:
    """Підставний солвер: запам'ятовує аргументи й який метод викликали."""

    last = None

    def __init__(self, simulator, diameters, **kwargs):
        self.simulator = simulator
        self.diameters = diameters
        self.kwargs = kwargs
        self.called = None
        self.ctx = type("Ctx", (), {"sim_count": 123})()
        self.history = [(123, 6.0e6)]
        _SpySolver.last = self

    def solve_standalone(self, **_):
        self.called = "solve_standalone"
        return [0.4, 0.5, 0.6]

    def solve_fast(self, **_):
        self.called = "solve_fast"
        return [0.4, 0.5, 0.6]


@pytest.fixture
def spied_runner(monkeypatch, sim):
    """Раннер із підставним солвером і фальшивим симулятором."""
    import runner as runner_module

    monkeypatch.setattr(runner_module, "AnalyticalSolver", _SpySolver)

    def build(**request_kw):
        r = OptimizationRunner(make_request(**request_kw))
        r.config = sim.config
        r.simulator = sim
        return r

    return build


def test_sequential_run_asks_for_one_worker_and_no_time_limit(spied_runner):
    spied_runner(sequential=True, v_opt=1.3)._single_run(1, "/tmp/logs")

    kw = _SpySolver.last.kwargs
    assert kw["n_workers"] == 1
    assert kw["time_limit_sec"] == float("inf")
    assert kw["pool"] is None
    assert kw["max_sims"] is None
    assert kw["log_dir"] == "/tmp/logs"
    assert _SpySolver.last.kwargs["v_opt"] == 1.3


def test_parallel_run_forwards_cores_budget_and_limit(spied_runner):
    spied_runner(cores=3, max_sims=250_000, time_limit=90)._single_run(2, "/tmp/logs")

    kw = _SpySolver.last.kwargs
    assert (kw["n_workers"], kw["max_sims"], kw["time_limit_sec"]) == (3, 250_000, 90)


def test_default_time_limit_stays_unset_so_the_solver_derives_it(spied_runner):
    """None означає «вивести з калібрування» — так було до етапу 5."""
    spied_runner()._single_run(1, "/tmp/logs")
    assert _SpySolver.last.kwargs["time_limit_sec"] is None


def test_fast_mode_dispatches_to_solve_fast(spied_runner):
    spied_runner(run_mode="fast_analytical")._single_run(1, "/tmp/logs")
    assert _SpySolver.last.called == "solve_fast"


def test_analytical_mode_dispatches_to_solve_standalone(spied_runner):
    spied_runner()._single_run(1, "/tmp/logs")
    assert _SpySolver.last.called == "solve_standalone"


def test_result_carries_stats_of_the_returned_solution(spied_runner):
    result = spied_runner()._single_run(7, "/tmp/logs")

    assert result.run_id == 7
    assert result.individual == [1, 2, 3]          # 0.4, 0.5, 0.6 -> індекси
    assert result.history == [{"evals": 123, "min_cost": 6.0e6}]
    # Вартість і тиск беруться з симулятора для повернутих діаметрів, а не
    # з того, що солвер сам про себе повідомив.
    assert result.cost == 470_000.0
    assert result.pressure == pytest.approx(76.273, abs=1e-3)
    assert result.feasible is True


def test_empty_solution_yields_no_result(spied_runner, monkeypatch):
    monkeypatch.setattr(_SpySolver, "solve_standalone", lambda *_, **__: [])
    assert spied_runner()._single_run(1, "/tmp/logs") is None
