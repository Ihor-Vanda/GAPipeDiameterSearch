import time
import networkx as nx
from .cache import LRUCache

class SolverContext:
    def __init__(self, simulator, available_diameters, v_opt=1.0):
        self.simulator = simulator
        self.diameters = sorted(available_diameters)
        self.max_d_idx = len(self.diameters) - 1
        self.v_opt = v_opt
        self.num_pipes = simulator.n_variables
        
        self.comp_name_to_idx = {name: i for i, name in enumerate(simulator.component_names)}
        
        self.sim_cache = LRUCache(50000)
        self.heuristic_cache = LRUCache(50000)
        
        self.sim_count = 0
        
        self.base_G_flow = nx.Graph()
        self.edge_to_pipe = {}
        self.node_to_pipes = {} 
        
        for u, v, k in self.simulator.graph.edges(keys=True):
            if k in self.comp_name_to_idx:
                idx = self.comp_name_to_idx[k]
                self.base_G_flow.add_edge(u, v)
                self.edge_to_pipe[(u, v)] = idx
                self.edge_to_pipe[(v, u)] = idx
                self.node_to_pipes.setdefault(u, []).append(idx)
                self.node_to_pipes.setdefault(v, []).append(idx)

        self.lengths = self.simulator.lengths
        self.costs_array = self.simulator.costs 

        diffs = [abs(c2-c1) for c1, c2 in zip(self.costs_array[:-1], self.costs_array[1:])]
        avg_cost_diff = sum(diffs) / len(diffs) if len(diffs) > 0 else 50
        avg_length = sum(self.lengths) / len(self.lengths) if len(self.lengths) > 0 else 100
        self.baseline_bonus = avg_cost_diff * avg_length * 0.8
        
        self.sim_speed = self._calibrate_simulator()
        print(f"[SolverContext] Hardware calibrated: ~{self.sim_speed:.0f} sims/sec")
        
    def is_ghost_solution(self, indices, cost):
        sig = tuple(indices)
        cached = self.sim_cache.get(sig)
        if cached is not None:
            _, real_p_min, _, _ = cached
            return real_p_min < self.simulator.config.h_min - 0.01
            
        try:
            result = self.simulator.get_stats(indices)  # без конвертації
            self.sim_cache.set(sig, result)
            return result[1] < self.simulator.config.h_min - 0.01
        except Exception as e:
            self.log(f"     [CRITICAL ERROR in Ghost Check]: {e}")
            return True

    def _calibrate_simulator(self):
        mid_idx = self.max_d_idx // 2
        test_idx = [mid_idx] * self.num_pipes
        
        for _ in range(3): 
            self.simulator.get_stats(test_idx) 
            
        start = time.time()
        for i in range(10):
            noisy_idx = list(test_idx)
            noisy_idx[i % self.num_pipes] = max(0, noisy_idx[i % self.num_pipes] - 1)
            self.simulator.get_stats(noisy_idx)
            
        elapsed = time.time() - start
        return 10.0 / elapsed if elapsed > 0 else 1000.0
    
    def log(self, message):
        print(f"   {message}", flush=True)

    def get_fingerprint(self, indices, known_cost=None):
        sol_tuple = tuple(indices)
        if known_cost is not None:
            cost_bucket = int(known_cost / 50) * 50 
            return (cost_bucket, sol_tuple)
        cost, _, _, _ = self.get_cached_stats(indices)
        return (int(cost / 50) * 50, sol_tuple)

    def get_cached_stats(self, indices):
        sig = tuple(indices)
        res = self.sim_cache.get(sig)
        if res is None:
            self.sim_count += 1 
            res = self.simulator.get_stats(indices)
            self.sim_cache.set(sig, res)
        return res
        
    def get_cached_heuristics(self, indices):
        sig = tuple(indices)
        res = self.heuristic_cache.get(sig)
        if res is None:
            res = self.simulator.get_heuristics(indices)
            self.heuristic_cache.set(sig, res)
        return res
    
    def get_dominant_path(self, indices, crit_node):
        try:
            for u, v in self.base_G_flow.edges():
                idx = self.edge_to_pipe[(u, v)]
                self.base_G_flow[u][v]['weight'] = 100.0 / (indices[idx] + 1)
                    
            best_path_nodes = []
            best_weight = float('inf')
            
            for source in self.simulator.sources:
                try:
                    path = nx.shortest_path(self.base_G_flow, source, crit_node, weight='weight')
                    path_weight = sum(self.base_G_flow[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    if path_weight < best_weight:
                        best_weight = path_weight
                        best_path_nodes = path
                except nx.NetworkXNoPath:
                    continue 
            
            if not best_path_nodes: return [], []
            path_pipes = [self.edge_to_pipe[(u, v)] for u, v in zip(best_path_nodes[:-1], best_path_nodes[1:])]
            return path_pipes, best_path_nodes
        except Exception:
            return [], []

    def get_lazy_periphery(self, indices):
        _, _, _, crit_node = self.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return indices, []
        path_pipes, _ = self.get_dominant_path(indices, crit_node)
        unit_losses = self.get_cached_heuristics(indices)
        threshold_idx = int(self.max_d_idx * 0.6)
        periphery_pipes = [i for i in range(self.num_pipes) if i not in path_pipes and indices[i] <= threshold_idx]
        periphery_pipes.sort(key=lambda i: unit_losses[i])
        return indices, periphery_pipes