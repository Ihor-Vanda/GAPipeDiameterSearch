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

from pipeopt.hydraulics.sim_stats import SimStats  # noqa: E402


@dataclass
class FakeConfig:
    """Мінімум того, що солвер читає з SolverConfig."""

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
    from pipeopt.solver.context import SolverContext

    return SolverContext(sim, sim.config.diameters_m, v_opt=1.0)


class LoopedSimulator(FakeSimulator):
    """Мережа з двома циклами і периферійною віткою.

    `FakeSimulator` — ланцюг, і цього довго було досить. Але рівно через
    відсутність циклів **три кіки не перевірялись по суті**: вони
    виходили на «нічого не знайдено» в перших рядках і повертали
    `failure()`, тому тест «кік не впав» проходив, нічого не перевіривши:

    * `loop_balancing_kick` — `nx.cycle_basis` на ланцюзі дає порожній
      список, вихід «No cycles found» (покриття 53 %);
    * `topological_inversion_kick` — у ланцюзі єдиний шлях до критичного
      вузла збігається з домінантним, альтернатив немає, вихід
      «All paths tabu» (покриття 73 %);
    * `peripheral_trim_kick` — домінантний шлях у ланцюзі проходить
      **усіма** трубами, тому `get_lazy_periphery` завжди повертає
      порожній список, вихід «No targets».

    Топологія (R — джерело, G — тупик):

        R --P1-- A --P2-- B --P3-- C
                 |        |        |
                P4       P5       P6
                 |        |        |
                 D --P7-- E --P8-- F --P9-- G

    Два незалежні цикли (A-B-E-D і B-C-F-E) і тупикова вітка P9, яка
    гарантує непорожню периферію. 9 труб, 8 вузлів.

    Гідравліка спрощена так само, як у ланцюзі, і з тієї ж причини —
    тест мусить бути детермінованим і не тягнути EPANET:

    * витрата кожної труби **фіксована** (не залежить від діаметрів) і
      взята з кореневого дерева обходу: труба несе відбори всіх вузлів
      під собою. Двом трубам, що замикають цикли, призначено невелику
      транзитну витрату;
    * напір у вузлі = напір джерела мінус **найменша** сума втрат серед
      усіх шляхів до нього. Це та властивість, заради якої існують цикли:
      розширення труби на будь-якому зі шляхів піднімає напір, а поява
      дешевого обходу робить цикл корисним. Саме її й перевіряють
      `LOOP_BALANCE` і `TOPO-INV`.
    """

    # Підібрано так, щоб межа допустимості пройшла серединою таблиці
    # діаметрів: [0]*9 і [1]*9 недопустимі за тиском, [2]*9 і [3]*9
    # допустимі. Інакше `heal_network` у тестах або не мав би що робити,
    # або мусив би виводити все на максимум.
    C = 2.0e-2

    #: (ім'я, вузол-початок, вузол-кінець, витрата в частках відбору).
    #: Частки — розміри піддерев обходу від R; 1.5 у двох трубах, що
    #: замикають цикли, — транзит, який дерево обходу не описує.
    TOPOLOGY = (
        ("P1", "R", "A", 7.0),
        ("P2", "A", "B", 5.0),
        ("P3", "B", "C", 3.0),
        ("P4", "A", "D", 1.0),
        ("P5", "B", "E", 1.0),
        ("P6", "C", "F", 2.0),
        ("P7", "D", "E", 1.5),
        ("P8", "E", "F", 1.5),
        ("P9", "F", "G", 1.0),
    )

    def __init__(self, config=None, source_head=100.0, demand=0.03):
        super().__init__(n_pipes=len(self.TOPOLOGY), config=config,
                         source_head=source_head, demand=demand)

        self.component_names = [name for name, _, _, _ in self.TOPOLOGY]
        self.node_names = ["R", "A", "B", "C", "D", "E", "F", "G"]
        self.sources = ["R"]
        self.flow_shares = [share for _, _, _, share in self.TOPOLOGY]

        self.graph = nx.MultiGraph()
        for name, u, v, _ in self.TOPOLOGY:
            self.graph.add_edge(u, v, key=name)

    def _flows(self):
        return [share * self.demand for share in self.flow_shares]

    def _heads(self, indices):
        """Напір у кожному вузлі: найдешевший за втратами шлях від джерела."""
        losses = self._losses(indices)
        weighted = nx.Graph()
        for pipe_no, (name, u, v, _) in enumerate(self.TOPOLOGY):
            weighted.add_edge(u, v, weight=losses[pipe_no])
        drops = nx.single_source_dijkstra_path_length(weighted, "R", weight="weight")
        return {node: self.source_head - drop for node, drop in drops.items()}

    def get_stats(self, indices):
        self.sim_count += 1
        heads = self._heads(indices)
        junctions = [node for node in self.node_names if node != "R"]
        pressures = {node: heads[node] for node in junctions}
        crit_node = min(pressures, key=pressures.get)
        return SimStats(self._cost(indices), pressures[crit_node],
                        max(pressures.values()), crit_node)


@pytest.fixture
def looped_sim():
    return LoopedSimulator()


@pytest.fixture
def looped_ctx(looped_sim):
    from pipeopt.solver.context import SolverContext

    return SolverContext(looped_sim, looped_sim.config.diameters_m, v_opt=1.0)


#: Модулі пакета `pipeopt/solver/island`. Список перевіряється самою
#: фікстурою — інакше вона тихо перестане сканувати те, що переїхало.
ISLAND_MODULES = ("__init__.py", "beam.py", "diversification.py",
                  "exchange.py", "loop.py", "state.py")


@pytest.fixture(scope="session")
def island_source():
    """Весь текст пакета `island` одним рядком.

    Кілька тестів шукають у вихідному тексті дублювання констант — колись
    вони брали `inspect.getsource(IslandWorker)`, і це працювало, поки
    клас був один файл на 1283 рядки. Після розділення на домішки той
    самий виклик повертає **тільки** тіло зібраного класу, тобто десяток
    рядків: перевірка проходила б завжди, нічого не перевіряючи.

    Точно так само вже ламалась перевірка контракту кіків після
    розділення `kicks.py` (етап 20), тому тут список модулів звіряється з
    диском **усередині фікстури**: розійшовся — падають усі тести, які
    нею користуються, а не мовчки зеленіють.
    """
    package = Path(__file__).resolve().parents[1] / "pipeopt" / "solver" / "island"
    on_disk = {path.name for path in package.glob("*.py")}

    assert on_disk == set(ISLAND_MODULES), (
        f"модулі пакета island змінились: на диску {sorted(on_disk)}, "
        f"скануються {sorted(ISLAND_MODULES)}")

    return "\n".join((package / name).read_text(encoding="utf-8")
                     for name in ISLAND_MODULES)
