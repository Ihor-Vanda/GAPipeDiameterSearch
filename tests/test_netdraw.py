"""Малювання мережі: шкала діаметрів і чотири джерела значень.

`netdraw` не залежить ні від tkinter, ні від EPANET (імпорт `wntr`
локальний у `NetworkGeometry.load`), тому тут перевіряється саме те, що в
чотирьох копіях було скопійовано дослівно: формула товщини, дискретна
шкала кольору і — головне — **порядок** значень.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import netdraw  # noqa: E402


# --- шкала діаметрів --------------------------------------------------

def test_each_size_gets_its_own_color_index():
    scale = netdraw.DiameterScale([0.3, 0.5, 0.3, 0.4])
    assert scale.unique == [0.3, 0.4, 0.5]
    assert scale.color_indices == [0, 2, 0, 1]


def test_width_grows_linearly_between_the_extremes():
    scale = netdraw.DiameterScale([0.2, 0.4, 0.6])
    assert scale.widths[0] == pytest.approx(1.0, abs=1e-4)
    assert scale.widths[1] == pytest.approx(3.0, abs=1e-4)
    assert scale.widths[2] == pytest.approx(5.0, abs=1e-4)


def test_equal_diameters_do_not_divide_by_zero():
    """Саме для цього в усіх чотирьох копіях стояло `+ 1e-6`."""
    scale = netdraw.DiameterScale([0.4, 0.4, 0.4])
    assert scale.widths == [pytest.approx(1.0)] * 3


def test_unknown_diameter_gets_the_base_width_and_first_color():
    scale = netdraw.DiameterScale([0.3, netdraw.UNKNOWN_DIAMETER, 0.6])
    assert scale.widths[1] == netdraw.BASE_WIDTH
    assert scale.color_indices[1] == 0
    assert scale.unique == [0.3, 0.6]          # невідома не потрапляє в легенду


def test_scale_without_any_diameter_reports_no_data():
    scale = netdraw.DiameterScale([netdraw.UNKNOWN_DIAMETER, 0.0])
    assert scale.has_data is False
    assert scale.unique == []


def test_colorbar_labels_are_the_sizes_themselves():
    fig, ax = plt.subplots()
    scale = netdraw.DiameterScale([0.3, 0.4064, 0.5])

    cbar = scale.attach_colorbar(fig, ax, "Діаметр (mm)")
    assert [t.get_text() for t in cbar.ax.get_yticklabels()] == ["0.3", "0.4064", "0.5"]
    plt.close(fig)


def test_colorbar_is_reused_rather_than_stacked():
    """Жива топологія оновлює легенду на кожному кадрі; якби вона
    створювала нову, вони накопичувались би збоку фігури."""
    fig, ax = plt.subplots()
    first = netdraw.DiameterScale([0.3, 0.5]).attach_colorbar(fig, ax, "d")

    again = netdraw.DiameterScale([0.3, 0.4, 0.5]).attach_colorbar(fig, ax, "d",
                                                                   existing=first)
    assert again is first
    assert [t.get_text() for t in again.ax.get_yticklabels()] == ["0.3", "0.4", "0.5"]
    plt.close(fig)


def test_no_colorbar_without_data():
    fig, ax = plt.subplots()
    assert netdraw.DiameterScale([-1.0]).attach_colorbar(fig, ax, "d") is None
    plt.close(fig)


# --- геометрія без wntr -----------------------------------------------

class FakePipe:
    def __init__(self, diameter):
        self.diameter = diameter


class FakeWn:
    """Мінімум інтерфейсу `WaterNetworkModel`, який читає `netdraw`.

    Труби навмисно перелічені НЕ в порядку ребер графа — саме на цій
    невідповідності розходились чотири копії.
    """

    def __init__(self, diameters=None):
        self.pipe_name_list = ["P1", "P2", "P3"]
        self.junction_name_list = ["J1", "J2", "SPARE"]
        self.reservoir_name_list = ["R"]
        self.tank_name_list = []
        self._pipes = {name: FakePipe(d) for name, d in
                       zip(self.pipe_name_list, diameters or [0.3, 0.4, 0.5])}

    def get_link(self, name):
        return self._pipes[name]


def make_geometry(diameters=None, edge_order=("P3", "P1", "P2")):
    wn = FakeWn(diameters)
    graph = nx.MultiGraph()
    chain = {"P1": ("R", "J1"), "P2": ("J1", "J2"), "P3": ("J2", "J1")}
    for key in edge_order:
        u, v = chain[key]
        graph.add_edge(u, v, key=key)
    pos = {n: (i, 0) for i, n in enumerate(["R", "J1", "J2"])}
    return netdraw.NetworkGeometry(wn, graph, pos, list(edge_order))


def test_node_groups_skip_names_absent_from_the_graph():
    """`SPARE` є у списку вузлів, але не в графі — фільтр був лише в
    одному з чотирьох місць, решта впала б."""
    geometry = make_geometry()
    assert geometry.junctions == ["J1", "J2"]
    assert geometry.sources == ["R"]


# --- джерела діаметрів: усі в порядку ребер графа ----------------------

def test_indices_are_mapped_through_pipe_names_not_positions():
    geometry = make_geometry()
    table = [0.1, 0.2, 0.3, 0.4]

    # рішення в порядку pipe_name_list: P1->0.2, P2->0.4, P3->0.1
    diameters = netdraw.diameters_from_indices(geometry, [1, 3, 0], diameters_m=table)

    # а повертається в порядку ребер графа: P3, P1, P2
    assert diameters == [0.1, 0.2, 0.4]


def test_raw_millimetres_are_converted_when_no_metre_table_is_given():
    geometry = make_geometry()
    diameters = netdraw.diameters_from_indices(geometry, [0, 1, 2],
                                               diameters_raw=[300.0, 400.0, 500.0])
    assert diameters == [0.5, 0.3, 0.4]


def test_index_outside_the_table_is_unknown_rather_than_an_error():
    geometry = make_geometry()
    diameters = netdraw.diameters_from_indices(geometry, [0, 99, 0],
                                               diameters_m=[0.3, 0.4])
    assert diameters == [0.3, 0.3, netdraw.UNKNOWN_DIAMETER]


def test_live_solution_is_mapped_through_the_size_table():
    geometry = make_geometry()
    diameters = netdraw.diameters_from_solution(geometry, [2, 0, 1],
                                                table=[0.3, 0.4, 0.5])
    assert diameters == [0.4, 0.5, 0.3]


def test_live_solution_without_a_table_falls_back_to_indices():
    """Кольори все одно мусять розрізняти типорозміри."""
    geometry = make_geometry()
    assert netdraw.diameters_from_solution(geometry, [2, 0, 1]) == [2.0, 3.0, 1.0]


def test_solution_shorter_than_the_network_marks_the_rest_unknown():
    geometry = make_geometry()
    diameters = netdraw.diameters_from_solution(geometry, [1], table=[0.3, 0.4])
    assert diameters == [netdraw.UNKNOWN_DIAMETER, 0.4, netdraw.UNKNOWN_DIAMETER]


def test_inp_diameters_in_metres_are_shown_in_millimetres():
    geometry = make_geometry(diameters=[0.3, 0.4, 0.5])
    assert netdraw.diameters_from_inp(geometry) == [500.0, 300.0, 400.0]


def test_inp_diameters_already_in_millimetres_are_left_alone():
    geometry = make_geometry(diameters=[300.0, 400.0, 500.0])
    assert netdraw.diameters_from_inp(geometry) == [500.0, 300.0, 400.0]


def test_csv_column_is_found_by_name_not_by_position():
    geometry = make_geometry()
    frame = pd.DataFrame({"Pipe": ["P1", "P2", "P3"],
                          "Діаметр (mm)": [300.0, 400.0, 500.0],
                          "Cost": [1.0, 2.0, 3.0]})

    assert netdraw.diameters_from_table(geometry, frame) == [500.0, 300.0, 400.0]


def test_csv_integers_are_read_as_indices_into_the_size_table():
    geometry = make_geometry()
    frame = pd.DataFrame({"Pipe": ["P1", "P2", "P3"], "diameter": [0, 2, 1]})

    diameters = netdraw.diameters_from_table(geometry, frame, table=[0.3, 0.4, 0.5])
    assert diameters == [0.4, 0.3, 0.5]


def test_pipe_missing_from_the_csv_is_unknown():
    geometry = make_geometry()
    frame = pd.DataFrame({"Pipe": ["P1"], "diameter": [350.0]})

    diameters = netdraw.diameters_from_table(geometry, frame)
    assert diameters == [netdraw.UNKNOWN_DIAMETER, 350.0, netdraw.UNKNOWN_DIAMETER]


# --- малювання --------------------------------------------------------

def test_draw_puts_nodes_over_pipes_by_default():
    fig, ax = plt.subplots()
    netdraw.draw(ax, make_geometry(), [0.3, 0.4, 0.5])

    kinds = [type(c).__name__ for c in ax.collections]
    assert kinds.index("LineCollection") < kinds.index("PathCollection")
    plt.close(fig)


def test_report_draws_pipes_over_nodes():
    """Порядок PNG-звіту збережено дослівно (§19.2)."""
    fig, ax = plt.subplots()
    netdraw.draw(ax, make_geometry(), [0.3, 0.4, 0.5], nodes_on_top=False)

    kinds = [type(c).__name__ for c in ax.collections]
    assert kinds.index("PathCollection") < kinds.index("LineCollection")
    plt.close(fig)


def test_draw_hides_the_axes_and_keeps_the_aspect():
    fig, ax = plt.subplots()
    netdraw.draw(ax, make_geometry(), [0.3, 0.4, 0.5])

    assert ax.get_aspect() == 1.0
    assert ax.axison is False
    plt.close(fig)
