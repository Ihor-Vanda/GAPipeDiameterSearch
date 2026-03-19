import numpy as np
from numba import njit

# @njit компілює цю функцію в машинний код під час першого виклику
@njit(fastmath=True)
def fast_hamming_distance(sol1, sol2):
    dist = 0
    for i in range(len(sol1)):
        if sol1[i] != sol2[i]:
            dist += 1
    return dist

@njit(fastmath=True)
def fast_avg_hamming(pool_matrix):
    # pool_matrix - це 2D NumPy масив: розмірність (N_рішень, M_труб)
    n = pool_matrix.shape[0]
    if n < 2: return 0.0
    
    total_dist = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += fast_hamming_distance(pool_matrix[i], pool_matrix[j])
            pairs += 1
    return total_dist / pairs

@njit
def fast_dijkstra(num_nodes, sources, target, indptr, indices, weights):
    # indptr, indices, weights - це представлення графа у форматі CSR (Compressed Sparse Row)
    dist = np.full(num_nodes, np.inf)
    parent = np.full(num_nodes, -1)
    
    # Мульти-старт з усіх джерел (sources)
    for s in sources:
        dist[s] = 0.0
        
    visited = np.zeros(num_nodes, dtype=np.bool_)
    
    for _ in range(num_nodes):
        # Шукаємо невідвіданий вузол з мінімальною відстанню
        u = -1
        min_d = np.inf
        for i in range(num_nodes):
            if not visited[i] and dist[i] < min_d:
                min_d = dist[i]
                u = i
                
        if u == -1 or u == target:
            break
            
        visited[u] = True
        
        # Релаксація сусідів
        for edge_idx in range(indptr[u], indptr[u+1]):
            v = indices[edge_idx]
            w = weights[edge_idx]
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                
    # Відновлення шляху
    if dist[target] == np.inf:
        # Якщо шлях не знайдено (недосяжний вузол)
        return [np.int32(x) for x in range(0)] # Порожній типізований список
        
    path = []
    curr = target
    while curr != -1:
        path.append(np.int32(curr))
        if dist[curr] == 0: 
            break
        curr = parent[curr]
        
    path_rev = [path[i] for i in range(len(path)-1, -1, -1)]
    return path_rev

@njit
def fast_crossover(sol1, sol2, p_mine):
    n = len(sol1)
    child = np.empty(n, dtype=np.int32)
    # Numba вміє генерувати масиви випадкових чисел на швидкості C
    rand_vals = np.random.random(n) 
    locked_count = 0
    
    for i in range(n):
        if sol1[i] == sol2[i]:
            child[i] = sol1[i]
        else:
            if rand_vals[i] < p_mine:
                child[i] = sol1[i]
            else:
                child[i] = sol2[i]
            locked_count += 1
            
    return child