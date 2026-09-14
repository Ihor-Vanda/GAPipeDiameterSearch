"""Малювання мережі: геометрія, шкала діаметрів, вузли, ребра.

До цього модуля та сама картинка малювалась у чотирьох місцях:

* `plot.py::plot_network_map` — PNG-звіт після прогону;
* `gui.py::update_live_topology` — жива топологія під час пошуку;
* `gui.py::_draw_basic_topology` — початкова мережа з діаметрами з `.inp`;
* `gui.py` (перегляд результатів) — карта, відновлена з `solution.csv`.

Спільного в них — усе, крім того, **звідки беруться діаметри**: побудова
графа й координат, дискретна шкала `turbo`, формула товщини
`1 + 4 * (d - min) / (max - min + 1e-6)`, кольорова легенда з підписами
`f"{d:g}"`, розміри й кольори вузлів. Формула товщини була скопійована
чотири рази дослівно, включно з `1e-6`.

Тут лишається і спільне малювання, і чотири **джерела діаметрів** —
`diameters_from_*`. Кожне з них відповідає на одне питання: який діаметр
у кожної труби **в порядку ребер графа**. Порядок тут — половина
складності: вектор рішення індексується порядком `pipe_name_list`, а
`networkx` малює ребра у своєму, і всі чотири копії розв'язували цю
невідповідність по-своєму.

Залежність від `wntr` локальна в `NetworkGeometry.load`, тому шкалу можна
тестувати без EPANET.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx

import solution_table

#: Труби без відомого діаметра позначаються від'ємним значенням і малюються
#: тонкою лінією базового кольору.
UNKNOWN_DIAMETER = -1.0

JUNCTION_COLOR = "#95a5a6"
RESERVOIR_COLOR = "#3498db"
TANK_COLOR = "#2ecc71"
NO_DATA_EDGE_COLOR = "#cccccc"

JUNCTION_SIZE = 15
#: PNG-звіт друкується у 300 dpi на 14×14 дюймів, тому маркери джерел там
#: більші, ніж у вікні GUI.
SOURCE_SIZE_REPORT = 150
SOURCE_SIZE_SCREEN = 100

BASE_WIDTH = 1.0
WIDTH_SPAN = 4.0
COLORMAP = "turbo"


@dataclass(frozen=True)
class NetworkGeometry:
    """Мережа, готова до малювання: граф, координати, порядок труб."""

    wn: object
    graph: object
    pos: dict
    #: Порядок труб у графі — саме в цьому порядку `networkx` малює ребра.
    #: Усі `diameters_from_*` повертають значення в цьому ж порядку.
    edge_pipe_names: list

    @classmethod
    def load(cls, inp_file):
        import warnings

        import wntr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(inp_file)

        graph = wn.get_graph()

        # GUI діставав координати через `query_node_attribute('coordinates')`
        # і не мав запасного варіанту: на мережі без координат малювання
        # падало, і вікно тихо показувало готовий PNG замість графа.
        # Запасний layout із `plot.py` тепер спільний.
        pos = {}
        for name in wn.node_name_list:
            node = wn.get_node(name)
            coords = getattr(node, "coordinates", None)
            if coords is not None:
                pos[name] = coords
        if not pos:
            pos = nx.kamada_kawai_layout(graph)

        edge_pipe_names = [key for _, _, key in graph.edges(keys=True)]
        return cls(wn, graph, pos, edge_pipe_names)

    # --- групи вузлів -----------------------------------------------------
    #
    # Фільтр `if n in graph` був лише в одному з чотирьох місць. Він нічого
    # не змінює, коли всі вузли є в графі, і рятує від помилки, коли ні.

    def _present(self, names):
        return [n for n in names if n in self.graph]

    @property
    def junctions(self):
        return self._present(self.wn.junction_name_list)

    @property
    def reservoirs(self):
        return self._present(self.wn.reservoir_name_list)

    @property
    def tanks(self):
        return self._present(self.wn.tank_name_list)

    @property
    def sources(self):
        return self.reservoirs + self.tanks

    def pipe_order(self):
        """Порядок труб у `wn.pipe_name_list` — порядок вектора рішення."""
        return list(self.wn.pipe_name_list)


class DiameterScale:
    """Дискретна шкала кольору й товщини за списком діаметрів.

    Кожен унікальний діаметр отримує власний колір з `turbo`, тому легенда
    читається як перелік типорозмірів, а не як континуум. Товщина —
    лінійна від найменшого до найбільшого діаметра.

    `values` — діаметри в тому ж порядку, в якому малюються ребра;
    від'ємне значення означає «діаметр невідомий».
    """

    def __init__(self, values):
        self.values = list(values)
        valid = [d for d in self.values if d > 0]

        self.has_data = bool(valid)
        self.unique = sorted(set(valid))
        self.min = min(valid) if valid else BASE_WIDTH
        self.max = max(valid) if valid else BASE_WIDTH

        self.cmap = plt.get_cmap(COLORMAP, max(1, len(self.unique)))
        color_of = {d: i for i, d in enumerate(self.unique)}
        self.color_indices = [color_of.get(d, 0) if d > 0 else 0 for d in self.values]
        self.widths = [self._width(d) for d in self.values]

    def _width(self, d):
        if d <= 0:
            return BASE_WIDTH
        # `1e-6` тримає знаменник ненульовим, коли всі діаметри однакові.
        return BASE_WIDTH + WIDTH_SPAN * (d - self.min) / (self.max - self.min + 1e-6)

    def _mappable(self):
        sm = plt.cm.ScalarMappable(
            cmap=self.cmap,
            norm=plt.Normalize(vmin=-0.5, vmax=len(self.unique) - 0.5),
        )
        sm.set_array([])
        return sm

    def attach_colorbar(self, fig, ax, label, existing=None, pad=None):
        """Створити або оновити легенду шкали. Повертає colorbar або None.

        `existing` — вже створена легенда (жива топологія GUI перемальовує
        її на кожному оновленні, а не створює нову: інакше вони
        накопичувались би збоку фігури).
        """
        if not self.has_data:
            return None

        sm = self._mappable()
        ticks = range(len(self.unique))
        labels = [f"{d:g}" for d in self.unique]

        if existing is not None:
            existing.update_normal(sm)
            existing.set_ticks(ticks)
            existing.ax.set_yticklabels(labels)
            return existing

        kwargs = {"shrink": 0.7}
        if pad is not None:
            kwargs["pad"] = pad
        cbar = fig.colorbar(sm, ax=ax, label=label, ticks=ticks, **kwargs)
        cbar.ax.set_yticklabels(labels)
        return cbar


def draw_edges(ax, geometry, scale):
    """Намалювати труби. Без жодного відомого діаметра — сірим одним тоном."""
    if not scale.has_data:
        nx.draw_networkx_edges(geometry.graph, geometry.pos, ax=ax,
                               edge_color=NO_DATA_EDGE_COLOR, alpha=0.8, arrows=False)
        return

    nx.draw_networkx_edges(geometry.graph, geometry.pos, ax=ax,
                           edge_color=scale.color_indices, edge_cmap=scale.cmap,
                           width=scale.widths, arrows=False)


def draw_nodes(ax, geometry, source_size=SOURCE_SIZE_SCREEN, legend_labels=False):
    """Намалювати вузли: вузли-відбори, резервуари (квадрат), баки (трикутник).

    `legend_labels` додає підписи для `ax.legend()` — потрібні лише
    PNG-звіту, у вікні GUI легенди вузлів немає.
    """
    groups = (
        (geometry.junctions, JUNCTION_COLOR, "o", JUNCTION_SIZE, None),
        (geometry.reservoirs, RESERVOIR_COLOR, "s", source_size, "Джерело"),
        (geometry.tanks, TANK_COLOR, "^", source_size, "Бак"),
    )

    for names, color, shape, size, label in groups:
        if not names:
            continue
        kwargs = {"label": label} if legend_labels and label else {}
        nx.draw_networkx_nodes(geometry.graph, geometry.pos, nodelist=names, ax=ax,
                               node_size=size, node_color=color, node_shape=shape,
                               **kwargs)


def draw(ax, geometry, values, source_size=SOURCE_SIZE_SCREEN,
         legend_labels=False, nodes_on_top=True):
    """Уся картинка одним викликом. Повертає шкалу для легенди.

    `nodes_on_top` — порядок малювання. Чотири копії розходились саме тут:
    у PNG-звіті труби лягали поверх маркерів вузлів, у вікні GUI — навпаки.
    Параметр зберігає обидва варіанти дослівно; за замовчуванням маркери
    джерел видно (варіант GUI).
    """
    scale = DiameterScale(values)
    if nodes_on_top:
        draw_edges(ax, geometry, scale)
        draw_nodes(ax, geometry, source_size=source_size, legend_labels=legend_labels)
    else:
        draw_nodes(ax, geometry, source_size=source_size, legend_labels=legend_labels)
        draw_edges(ax, geometry, scale)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    return scale


# ======================================================================
# Джерела діаметрів
# ======================================================================
#
# Усі повертають список у порядку `geometry.edge_pipe_names`, тобто в тому
# самому, в якому `networkx` малює ребра. Труба без відомого діаметра
# отримує `UNKNOWN_DIAMETER`.


def diameters_from_indices(geometry, individual, diameters_m=None, diameters_raw=None):
    """Вектор рішення + таблиця типорозмірів (PNG-звіт після прогону)."""
    table = diameters_m if diameters_m else [d / 1000.0 for d in (diameters_raw or [])]
    by_pipe = {name: int(idx) for name, idx in zip(geometry.pipe_order(), individual)}

    diameters = []
    for pipe_name in geometry.edge_pipe_names:
        idx = by_pipe.get(pipe_name)
        if idx is None or not 0 <= idx < len(table):
            diameters.append(UNKNOWN_DIAMETER)
        else:
            diameters.append(float(table[idx]))
    return diameters


def diameters_from_solution(geometry, solution, table=None):
    """Живе рішення під час пошуку (жива топологія GUI).

    Без таблиці типорозмірів кольори все одно мусять розрізняти варіанти,
    тому за умовний діаметр беруться самі індекси.
    """
    by_pipe = {name: i for i, name in enumerate(geometry.pipe_order())}

    diameters = []
    for pipe_name in geometry.edge_pipe_names:
        idx = by_pipe.get(pipe_name)
        if idx is None or idx >= len(solution):
            diameters.append(UNKNOWN_DIAMETER)
            continue
        choice = int(solution[idx])
        if table and 0 <= choice < len(table):
            diameters.append(float(table[choice]))
        else:
            diameters.append(float(choice + 1))
    return diameters


def diameters_from_inp(geometry):
    """Діаметри самого `.inp` — початковий стан мережі, до оптимізації.

    Евристика «менше 5 — це метри» лишається як була: `wntr` нормалізує
    діаметри в метри, тому 0.3 означає 300 мм, а 300 означає 300 мм.
    """
    diameters = []
    for pipe_name in geometry.edge_pipe_names:
        try:
            d = geometry.wn.get_link(pipe_name).diameter
        except Exception:
            diameters.append(0.0)
            continue
        diameters.append(d * 1000 if d < 5 else d)
    return diameters


def diameters_from_table(geometry, frame, table=None):
    """Рішення, відновлене з `solution.csv` (перегляд архіву прогону).

    Стовпець і правило «індекс чи діаметр» беруться з `solution_table`:
    те саме рішення потрібне і таблиці в перегляді результатів, і раніше
    було написане двічі (§27.5).
    """
    column = solution_table.diameter_column(frame)
    by_pipe = dict(zip(frame.iloc[:, 0].astype(str), frame[column]))

    diameters = []
    for pipe_name in geometry.edge_pipe_names:
        raw = by_pipe.get(str(pipe_name))
        if raw is None:
            diameters.append(UNKNOWN_DIAMETER)
            continue
        diameters.append(float(solution_table.real_diameter(raw, table)))
    return diameters
