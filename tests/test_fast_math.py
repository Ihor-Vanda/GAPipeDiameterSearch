"""Скомпільовані ядра: відстань Гемінга, найкоротший шлях, схрещування.

Ці чотири функції були єдиним модулем солвера **без жодного тесту** (15 %
покриття — тільки рядки імпорту). Причина зрозуміла: `@njit` компілює їх
при першому виклику, і здається, що перевіряти там нічого. Насправді
перевіряти є що саме через компіляцію: numba приймає не будь-які типи, і
помилка в типі аргумента виглядає як падіння десь глибоко в солвері.

`fast_dijkstra` — найскладніша з них: шлях у форматі CSR, кілька джерел
одночасно, і вага ребра, обернена до діаметра (`100 / (d + 1)`), тобто
«найкоротший» шлях — це шлях **найбільшого опору**, яким і йде критичний
тиск (§`SolverContext.get_dominant_path`).
"""

import numpy as np
import pytest

from pipeopt.solver.fast_math import (
    fast_avg_hamming, fast_dijkstra, fast_hamming_distance,
)


def csr(num_nodes, edges):
    """Побудувати CSR так само, як це робить `SolverContext.__init__`.

    `edges` — список `(u, v)`; кожне ребро додається в обидва напрямки,
    бо мережа неорієнтована.
    """
    neighbours = {i: [] for i in range(num_nodes)}
    for pipe, (u, v) in enumerate(edges):
        neighbours[u].append((v, pipe))
        neighbours[v].append((u, pipe))

    indptr, indices, edge_pipe = [0], [], []
    for node in range(num_nodes):
        for v, pipe in neighbours[node]:
            indices.append(v)
            edge_pipe.append(pipe)
        indptr.append(len(indices))

    return (np.array(indptr, dtype=np.int32),
            np.array(indices, dtype=np.int32),
            np.array(edge_pipe, dtype=np.int32))


def path(num_nodes, edges, sources, target, weights):
    indptr, indices, edge_pipe = csr(num_nodes, edges)
    w = np.array([weights[p] for p in edge_pipe], dtype=np.float64)
    result = fast_dijkstra(num_nodes, np.array(sources, dtype=np.int32),
                           target, indptr, indices, w)
    return [int(x) for x in result]


# --- відстань Гемінга -------------------------------------------------

def test_identical_solutions_are_at_distance_zero():
    a = np.array([1, 2, 3], dtype=np.int32)
    assert fast_hamming_distance(a, a) == 0


def test_distance_counts_differing_positions():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([1, 9, 3, 9], dtype=np.int32)
    assert fast_hamming_distance(a, b) == 2


def test_distance_is_symmetric():
    a = np.array([0, 5, 5], dtype=np.int32)
    b = np.array([5, 5, 0], dtype=np.int32)
    assert fast_hamming_distance(a, b) == fast_hamming_distance(b, a)


def test_distance_ignores_how_far_apart_the_values_are():
    """Гемінг рахує «інше чи те саме», не «наскільки інше» — на цьому
    тримається табу басейнів: рішення з одним іншим типорозміром є
    сусіднім, хай навіть діаметр змінився з найменшого на найбільший."""
    base = np.array([0, 0, 0], dtype=np.int32)
    near = np.array([0, 0, 1], dtype=np.int32)
    far = np.array([0, 0, 5], dtype=np.int32)
    assert fast_hamming_distance(base, near) == fast_hamming_distance(base, far)


# --- середня відстань у пулі ------------------------------------------

def test_a_pool_smaller_than_two_has_no_average_distance():
    assert fast_avg_hamming(np.zeros((0, 3), dtype=np.int32)) == 0.0
    assert fast_avg_hamming(np.zeros((1, 3), dtype=np.int32)) == 0.0


def test_a_pool_of_identical_solutions_has_zero_diversity():
    pool = np.ones((4, 5), dtype=np.int32)
    assert fast_avg_hamming(pool) == 0.0


def test_average_is_over_all_pairs_not_over_members():
    """Три рішення — три пари. Відстані 1, 1 і 2, отже середнє 4/3."""
    pool = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)
    assert fast_avg_hamming(pool) == pytest.approx(4 / 3)


def test_maximum_diversity_is_the_solution_length():
    pool = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    assert fast_avg_hamming(pool) == 3.0


# --- найкоротший шлях у CSR -------------------------------------------

def test_a_chain_gives_the_only_path():
    #  0 - 1 - 2 - 3
    result = path(4, [(0, 1), (1, 2), (2, 3)], [0], 3, [1.0, 1.0, 1.0])
    assert result == [0, 1, 2, 3]


def test_the_cheaper_of_two_routes_wins():
    #  0 - 1 - 3  (вага 1 + 1)   проти   0 - 2 - 3  (вага 5 + 5)
    edges = [(0, 1), (1, 3), (0, 2), (2, 3)]
    assert path(4, edges, [0], 3, [1.0, 1.0, 5.0, 5.0]) == [0, 1, 3]
    assert path(4, edges, [0], 3, [5.0, 5.0, 1.0, 1.0]) == [0, 2, 3]


def test_weights_decide_not_the_number_of_hops():
    """У солвері вага — `100 / (діаметр + 1)`, тому шлях із трьох тонких
    труб «коротший» за одну товсту: саме він і визначає критичний тиск."""
    #  довгий обхід 0-1-2-3 тонкими проти прямого 0-3 товстою
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    thin, thick = 1.0, 100.0
    assert path(4, edges, [0], 3, [thin, thin, thin, thick]) == [0, 1, 2, 3]
    assert path(4, edges, [0], 3, [thick, thick, thick, thin]) == [0, 3]


def test_the_nearest_of_several_sources_is_used():
    """Мережа з двома резервуарами: шлях мусить починатись у тому, від
    якого дешевше, а не в першому за списком."""
    #  0 - 2 (вага 9),  1 - 2 (вага 1);  джерела — 0 і 1
    result = path(3, [(0, 2), (1, 2)], [0, 1], 2, [9.0, 1.0])
    assert result == [1, 2]


def test_an_unreachable_target_gives_an_empty_path():
    #  0 - 1   і окремий вузол 2
    assert path(3, [(0, 1)], [0], 2, [1.0]) == []


def test_a_source_that_is_the_target_gives_a_single_node():
    assert path(2, [(0, 1)], [0], 0, [1.0]) == [0]


def test_the_path_starts_at_a_source_and_ends_at_the_target():
    """Інваріант, на який спирається `get_dominant_path`: він перетворює
    пари сусідніх вузлів у труби, тому порядок мусить бути від джерела."""
    result = path(5, [(0, 1), (1, 2), (2, 3), (3, 4)], [0], 4,
                  [1.0, 2.0, 3.0, 4.0])
    assert result[0] == 0
    assert result[-1] == 4
    assert len(result) == 5

# Тести `fast_crossover` прибрані разом із самою функцією: її не
# викликав жоден робочий шлях, і лише ці тести тримали її живою
# (та створювали враження 100 % покриття `fast_math.py`).
