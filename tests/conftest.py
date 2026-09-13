"""Спільні фікстури для тестів.

Ключова ідея: `FakeSimulator` реалізує той самий інтерфейс, що й
`WaterSimulator`, але без EPANET і без файлів — проста послідовна мережа
з аналітичною формулою втрат. Це робить придатними до юніт-тестування
`SolverContext`, `SolutionPool`, `LocalSearch` і `KickStrategies`:
тести виконуються за мілісекунди й не залежать від wntr, numba чи .inp.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_stats import SimStats  # noqa: E402


@dataclass
class FakeConfig:
    """Мінімум того, що солвер читає з GAConfig."""

    h_min: float = 30.0
    diameters_m: list = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6])
    diameters_raw: list = field(default_factory=lambda: [300.0, 400.0, 500.0, 600.0])
    costs: dict = field(default_factory=lambda: {300.0: 50.0, 400.0: 90.0, 500.0: 150.0, 600.0: 230.0})
    unit_system: str = "mm"
    run_mode: str = "analytical"
    v_opt: float = 1.0
    cost_file: str = "<fake>"


class FakeSimulator:
    """Послідовна мережа: R -> J1 -> ... -> Jn.

    Втрати в трубі i: C * L_i * Q_i^1.85 / d_i^4.87 (форма Хазена-Вільямса).
    Витрата в трубі i — сума відборів усіх вузлів нижче за течією, тому
    головні труби завантажені більше за периферійні. Тиск монотонно спадає
    вздовж ланцюга, отже критичний вузол — завжди останній.

    Модель детермінована й не має стану, тому будь-який тест повторюваний.
    """

    # Підібрано так, щоб обмеження реально працювало: рішення «усі мінімальні
    # діаметри» недопустиме за тиском, «усі максимальні» — допустиме із запасом.
    C = 5.0e-3

    def __init__(self, n_pipes=6, config=None, source_head=100.0, demand=0.03):
        self.config = config or FakeConfig()
        self.n_variables = n_pipes
        self.component_names = [f"P{i + 1}" for i in range(n_pipes)]
        self.lengths = [1000.0] * n_pipes
        self.costs = [self.config.costs[d] for d in self.config.diameters_raw]
        self.diams_m = self.config.diameters_m
        self.n_options = len(self.diams_m)
        self.source_head = source_head
        self.demand = demand
        self.sim_count = 0

        self.node_names = ["R"] + [f"J{i + 1}" for i in range(n_pipes)]
        self.sources = ["R"]

        self.graph = nx.MultiGraph()
        for i, name in enumerate(self.component_names):
            self.graph.add_edge(self.node_names[i], self.node_names[i + 1], key=name)

    # --- внутрішня гідравліка ---------------------------------------------

    def _flows(self):
        """Витрата в трубі i = відбір усіх вузлів нижче за течією."""
        n = self.n_variables
        return [(n - i) * self.demand for i in range(n)]

    def _losses(self, indices):
        flows = self._flows()
        losses = []
        for i, d_idx in enumerate(indices):
            d = self.diams_m[max(0, min(int(d_idx), self.n_options - 1))]
            losses.append(self.C * self.lengths[i] * abs(flows[i]) ** 1.85 / d ** 4.87)
        return losses

    def _cost(self, indices):
        total = 0.0
        for i, d_idx in enumerate(indices):
            d_idx = max(0, min(int(d_idx), self.n_options - 1))
            total += self.lengths[i] * self.costs[d_idx]
        return total

    # --- інтерфейс, який використовує солвер ------------------------------

    def get_stats(self, indices):
        self.sim_count += 1
        losses = self._losses(indices)
        head = self.source_head
        pressures = []
        for loss in losses:
            head -= loss
            pressures.append(head)
        min_p, max_p = min(pressures), max(pressures)
        crit_node = self.node_names[pressures.index(min_p) + 1]
        return SimStats(self._cost(indices), min_p, max_p, crit_node)

    def get_heuristics(self, indices):
        return [loss / L for loss, L in zip(self._losses(indices), self.lengths)]

    def get_hydraulic_state(self, real_diameters):
        idx = [min(range(self.n_options), key=lambda j: abs(self.diams_m[j] - d))
               for d in real_diameters]
        st = self.get_stats(idx)
        return self._flows(), st.is_feasible(self.config.h_min)


@pytest.fixture
def sim():
    return FakeSimulator()


@pytest.fixture
def ctx(sim):
    from analytical_solver.context import SolverContext

    return SolverContext(sim, sim.config.diameters_m, v_opt=1.0)
