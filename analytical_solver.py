import math
import random
import time
import bisect
import numpy as np
import itertools
import networkx as nx
import multiprocessing
from collections import OrderedDict

# =============================================================================
# 0. UTILS (LRU Cache)
# =============================================================================
class LRUCache:
    def __init__(self, maxsize):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        
    def get(self, key):
        if key not in self.cache: return None
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def set(self, key, value):
        if key in self.cache: 
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

# =============================================================================
# 1. SHARED CONTEXT (INDEX-ONLY PARADIGM)
# =============================================================================
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

# =============================================================================
# 2. SOLUTION POOL MANAGER 
# =============================================================================
class SolutionPool:
    def __init__(self, ctx):
        self.ctx = ctx
        self.active_pool = []
        self.tabu_fingerprints = {} # 🔴 Тепер це словник {fingerprint: round_added}
        self.kick_tabu_set = set()
        self.basin_tabu = set() 
        self.current_round = 0
        
    def clear_all(self):
        self.active_pool.clear()
        self.tabu_fingerprints.clear()
        self.kick_tabu_set.clear()
        self.basin_tabu.clear()

    def get_basin_signature(self, indices):
        """Створює унікальний підпис долини на основі гідравлічного скелета"""
        n = self.ctx.num_pipes
        top_k = max(3, n // 30)
        
        # Визначаємо структурно найважливіші труби для цього рішення
        unit_losses = self.ctx.get_cached_heuristics(indices)
        top_pipes = sorted(range(n), key=lambda i: unit_losses[i], reverse=True)[:top_k]
        
        return tuple(indices[p] for p in sorted(top_pipes))

    def is_basin_tabu(self, indices):
        return self.get_basin_signature(indices) in self.basin_tabu

    def add_basin_to_tabu(self, indices):
        self.basin_tabu.add(self.get_basin_signature(indices))

    def add_to_tabu(self, indices, cost):
        fp = self.ctx.get_fingerprint(indices, cost)
        self.tabu_fingerprints[fp] = self.current_round

    def is_tabu(self, indices, cost, tenure=80):
        fp = self.ctx.get_fingerprint(indices, cost)
        added_at = self.tabu_fingerprints.get(fp)
        if added_at is None: 
            return False
        # 🔴 Рішення забувається через 80 раундів
        return (self.current_round - added_at) < tenure
    
    def hamming_distance(self, sol1, sol2):
        return sum(1 for a, b in zip(sol1, sol2) if a != b)

# =============================================================================
# 3. LOCAL SEARCH ENGINE (WORKS ONLY WITH INDICES)
# =============================================================================
class LocalSearch:
    def __init__(self, ctx):
        self.ctx = ctx
        
    def get_high_impact_pipes(self, indices, top_k):
        unit_losses = self.ctx.get_cached_heuristics(indices)
        impact_scores = []
        for i in range(self.ctx.num_pipes):
            d_idx = indices[i]
            if d_idx > 0:
                save_potential = self.ctx.lengths[i] * (self.ctx.costs_array[d_idx] - self.ctx.costs_array[d_idx-1])
            else:
                save_potential = 0
                
            risk = max(unit_losses[i], 1e-5)
            impact = save_potential * (1.0 + risk)
            impact_scores.append((i, impact))
            
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in impact_scores[:top_k]]

    def gradient_squeeze(self, indices, locked_pipes=None, max_passes=None, quick_mode=False, dyn_bonus=None):
        locked_pipes = locked_pipes or set()
        current_indices = list(indices)
        improved = True
        passes = 0
        active_indices = [i for i in range(self.ctx.num_pipes) if i not in locked_pipes]
        
        cost, p_min, _, _ = self.ctx.get_cached_stats(current_indices)
        bonus = dyn_bonus if dyn_bonus is not None else self.ctx.baseline_bonus 
        
        while improved:
            improved = False
            passes += 1
            if max_passes and passes > max_passes: break
            
            unit_losses = self.ctx.get_cached_heuristics(current_indices)
            savings_candidates = []
            
            noise_threshold = max(1.0, bonus * 0.01)
            
            for i in active_indices:
                if quick_mode and unit_losses[i] > 0.05: continue
                c_idx = current_indices[i]
                if c_idx > 0:
                    dollar_save = self.ctx.lengths[i] * (self.ctx.costs_array[c_idx] - self.ctx.costs_array[c_idx - 1])
                    if dollar_save < noise_threshold: continue
                    
                    risk = unit_losses[i] + 1e-9 
                    
                    score = dollar_save / risk
                    savings_candidates.append((i, score, dollar_save))
            
            if not savings_candidates: break
            savings_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates_to_try = savings_candidates[:5 if quick_mode else 20]
            batch_count = 0
            
            for idx, _, save in candidates_to_try:
                c_idx = current_indices[idx] 
                test_indices = list(current_indices)
                test_indices[idx] = c_idx - 1
                
                _, t_p, t_feas, t_crit = self.ctx.get_cached_stats(test_indices)
                
                if t_feas and t_p >= self.ctx.simulator.config.h_min:
                    current_indices = test_indices
                    p_min = t_p
                    improved = True
                    batch_count += 1
                    if batch_count >= 5: break
                
                elif not quick_mode and save > (bonus * 0.5) and t_crit is not None:
                    neighbors = [i for i in self.ctx.node_to_pipes.get(t_crit, []) if i != idx and i not in locked_pipes]
                    best_fix, best_net_save, best_p = None, 0, p_min
                    
                    for n_idx in neighbors:
                        n_curr_d = test_indices[n_idx]
                        if n_curr_d < self.ctx.max_d_idx:
                            fix_indices = list(test_indices)
                            fix_indices[n_idx] = n_curr_d + 1 
                            expansion_cost = self.ctx.lengths[n_idx] * (self.ctx.costs_array[n_curr_d + 1] - self.ctx.costs_array[n_curr_d])
                            net_save = save - expansion_cost
                            
                            if net_save > (bonus * 0.5): 
                                _, f_p, f_feas, _ = self.ctx.get_cached_stats(fix_indices)
                                if f_feas and f_p >= self.ctx.simulator.config.h_min:
                                    if net_save > best_net_save:
                                        best_net_save, best_fix, best_p = net_save, fix_indices, f_p

                    if best_fix:
                        current_indices = best_fix
                        p_min = best_p
                        improved = True
                        batch_count += 1
                        if batch_count >= 5: break

        return current_indices

    def heal_network(self, indices, locked_pipes):
        kicked = list(indices)
        boosts = 0
        for _ in range(40): 
            _, t_p, t_feas, crit_node = self.ctx.get_cached_stats(kicked)
            if t_feas and t_p >= self.ctx.simulator.config.h_min: return kicked, True, boosts
            
            path_pipes, _ = self.ctx.get_dominant_path(kicked, crit_node)
            if not path_pipes: return kicked, False, boosts
            
            unit_losses = self.ctx.get_cached_heuristics(kicked)
            worst_pipe, max_loss = -1, -1.0
            
            for idx in path_pipes:
                if idx in locked_pipes: continue 
                if kicked[idx] < self.ctx.max_d_idx: 
                    if unit_losses[idx] > max_loss:
                        max_loss, worst_pipe = unit_losses[idx], idx
            
            if worst_pipe != -1:
                kicked[worst_pipe] += 1
                boosts += 1
            else: break
        return kicked, False, boosts

    def swap_search(self, indices, dyn_bonus):
        indices_copy = list(indices)
        best_cost, best_p, _, crit_node = self.ctx.get_cached_stats(indices_copy)
        
        if not crit_node or crit_node == "ERR":
            return indices_copy
            
        path_pipes, _ = self.ctx.get_dominant_path(indices_copy, crit_node)
        unit_losses = self.ctx.get_cached_heuristics(indices_copy)
        lazy_pipes = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i])
        
        if not path_pipes or not lazy_pipes:
            return indices_copy

        up_limit = max(10, self.ctx.num_pipes // 5)
        down_limit = max(10, self.ctx.num_pipes // 3)
        
        if self.ctx.num_pipes > 200:
            down_limit = min(down_limit, 40)
        
        # --- 1. Single Swap (Safe down-sizing) ---
        for p in lazy_pipes[:down_limit]:
            if indices_copy[p] > 0:
                test_sol = list(indices_copy)
                test_sol[p] -= 1
                c, p_val, feas, _ = self.ctx.get_cached_stats(test_sol)
                if feas and p_val >= self.ctx.simulator.config.h_min:
                    score = c - ((p_val - self.ctx.simulator.config.h_min) * dyn_bonus)
                    if score < best_cost:
                        best_cost, indices_copy = score, test_sol
                        
        # --- 2. Double Swap (1 Up, 2 Down) ---
        if self.ctx.num_pipes <= 200:
            for up_pipe in path_pipes[-int(up_limit*0.7):]:
                for d1, d2 in itertools.combinations(lazy_pipes[:int(down_limit*0.7)], 2):
                    if indices_copy[up_pipe] < self.ctx.max_d_idx and indices_copy[d1] > 0 and indices_copy[d2] > 0:
                        test_sol = list(indices_copy)
                        test_sol[up_pipe] += 1
                        test_sol[d1] -= 1
                        test_sol[d2] -= 1
                        
                        c, p_val, feas, _ = self.ctx.get_cached_stats(test_sol)
                        if feas and p_val >= self.ctx.simulator.config.h_min:
                            score = c - ((p_val - self.ctx.simulator.config.h_min) * dyn_bonus)
                            if score < best_cost:
                                best_cost, indices_copy = score, test_sol
                                
        return indices_copy

    def evaluate_candidate(self, base_indices, pipes_to_mod, mode, dyn_bonus):
        test_sol = list(base_indices)
        locked = set()
        valid = False
        for p_idx in pipes_to_mod:
            c_idx = test_sol[p_idx]
            if mode == "upgrade" and c_idx < self.ctx.max_d_idx:
                test_sol[p_idx] += 1
                locked.add(p_idx)
                valid = True
            elif mode == "downgrade" and c_idx > 0:
                test_sol[p_idx] -= 1
                valid = True
        
        if not valid: return float('inf'), -float('inf'), None
        
        squeezed_sol = self.gradient_squeeze(test_sol, locked_pipes=locked, max_passes=3, quick_mode=True, dyn_bonus=dyn_bonus)
        cost, p_min, _, _ = self.ctx.get_cached_stats(squeezed_sol)
        
        p_surplus = p_min - self.ctx.simulator.config.h_min
        if p_surplus < 0: return float('inf'), -float('inf'), None
        
        score = cost - (p_surplus * dyn_bonus)
        return score, cost, squeezed_sol

# =============================================================================
# 4. KICK STRATEGIES (INDEX-BASED + ITERATIVE DIJKSTRA)
# =============================================================================
class KickStrategies:
    def __init__(self, ctx, local_search):
        self.ctx = ctx
        self.ls = local_search

    def forcing_hand_kick(self, indices):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return indices, set(), ""

        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return indices, set(), ""
            
        kicked, locked = list(indices), set()
        limit = max(3, int(len(path_pipes) * 0.3))
        unit_losses = self.ctx.get_cached_heuristics(indices)
        
        worst_pipes = sorted(path_pipes, key=lambda i: unit_losses[i], reverse=True)[:limit]
        
        for idx in worst_pipes:
            if kicked[idx] < self.ctx.max_d_idx:
                kicked[idx] += 1
                locked.add(idx) 
        
        if not locked: return indices, set(), ""
        return kicked, locked, f"SHOCK: Upgraded {len(locked)} worst pipes on Critical Path"

    def upstream_bottleneck_kick(self, indices):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return None, None, ""
        
        best_candidate, reason = -1, ""
        for i in range(1, len(path_pipes)):
            prev_idx, curr_idx = path_pipes[i-1], path_pipes[i]
            d_prev, d_curr = indices[prev_idx], indices[curr_idx]
            if d_curr < d_prev and i < len(path_pipes) * 0.7:
                best_candidate, reason = curr_idx, f"Taper Violation (Idx{d_prev}->Idx{d_curr})"
                break
        
        if best_candidate == -1:
            unit_losses = self.ctx.get_cached_heuristics(indices)
            max_loss = -1.0
            limit = max(1, len(path_pipes) // 2)
            for idx in path_pipes[:limit]:
                if unit_losses[idx] > max_loss:
                    max_loss, best_candidate, reason = unit_losses[idx], idx, f"Max Upstream Loss ({max_loss:.4f})"

        if best_candidate == -1: return None, None, ""
        
        kicked, locked = list(indices), set()
        curr_d_idx = kicked[best_candidate] 
        target_idx = min(self.ctx.max_d_idx, curr_d_idx + 2)
        if best_candidate in path_pipes[:2]: target_idx = max(target_idx, self.ctx.max_d_idx - 1)

        if target_idx == curr_d_idx: return None, None, "" 

        kicked[best_candidate] = target_idx
        locked.add(best_candidate)
        return kicked, locked, f"BOTTLENECK: Boosted Pipe {best_candidate} ({reason})"

    def topological_inversion_kick(self, indices, tabu_set):
        """ПАТЧ: Iterative Penalized Dijkstra замість nx.shortest_simple_paths"""
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, "", None
        
        dom_pipes, dom_nodes = self.ctx.get_dominant_path(indices, crit_node)
        if not dom_nodes: return None, None, "", None
        
        source = dom_nodes[0]
        dom_edges = set(tuple(sorted((u, v))) for u, v in zip(dom_nodes[:-1], dom_nodes[1:]))
        
        target_capacity_idx = int(np.mean([indices[p] for p in dom_pipes])) if dom_pipes else self.ctx.max_d_idx
        
        temp_G = self.ctx.base_G_flow.copy()
        for u, v in temp_G.edges():
            idx = self.ctx.edge_to_pipe[(u, v)]
            temp_G[u][v]['weight'] = 100.0 / (indices[idx] + 1)
            
        candidates = []
        for _ in range(15):
            try:
                path = nx.shortest_path(temp_G, source, crit_node, weight='weight')
                if path != dom_nodes:
                    candidates.append(path)
                for u, v in zip(path[:-1], path[1:]):
                    temp_G[u][v]['weight'] *= 5.0
            except nx.NetworkXNoPath:
                break

        scored_paths = []
        for p_nodes in candidates:
            p_indices, overlap, length = [], 0, 0
            for u, v in zip(p_nodes[:-1], p_nodes[1:]):
                edge_key = tuple(sorted((u, v)))
                if edge_key in dom_edges: overlap += 1
                if (u, v) in self.ctx.edge_to_pipe:
                    p_indices.append(self.ctx.edge_to_pipe[(u, v)])
                    length += 1
            
            if length == 0: continue
            overlap_ratio = overlap / length
            signature = tuple(sorted(p_indices))
            if signature in tabu_set: continue
            scored_paths.append({'indices': p_indices, 'overlap': overlap_ratio, 'signature': signature})

        scored_paths.sort(key=lambda x: x['overlap'])
        if not scored_paths: return None, None, "All paths tabu", None

        best_alt = scored_paths[0]
        kicked, locked, push_count = list(indices), set(), 0
        
        for idx in best_alt['indices']:
            force_idx = max(target_capacity_idx, self.ctx.max_d_idx - 1) 
            force_idx = min(force_idx, self.ctx.max_d_idx)
            if kicked[idx] < force_idx:
                kicked[idx] = force_idx
                locked.add(idx)
                push_count += 1
            elif kicked[idx] == force_idx:
                 locked.add(idx)

        msg = f"Topo-Kick (Dijkstra): Boosting Alt Path (Overlap {best_alt['overlap']:.2f}, {push_count} boosted)"
        return kicked, locked, msg, best_alt['signature']

    def micro_trim_kick(self, indices):
        indices_copy, periphery_pipes = self.ctx.get_lazy_periphery(indices)
        if not periphery_pipes: return None, None, ""
        
        kicked, locked, shrunk_count = list(indices_copy), set(), 0
        limit = random.choice([1, 2, 3])
        
        for idx in periphery_pipes:
            if kicked[idx] > 0: 
                kicked[idx] -= 1
                locked.add(idx)
                shrunk_count += 1
            if shrunk_count >= limit: break
                
        if shrunk_count > 0:
            healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
            if is_feasible:
                return healed_sol, locked, f"FINISHER: Force-shrunk {shrunk_count} periphery pipes. Healed {boosts}x."
        return None, None, ""
    
    def sync_trim_kick(self, indices, dyn_bonus):
        indices_copy, periphery_pipes = self.ctx.get_lazy_periphery(indices)
        if not periphery_pipes: return None, None, ""
        candidates = []
        
        combo_limit = min(100, max(10, self.ctx.num_pipes // 3)) 
        all_pairs = list(itertools.combinations(periphery_pipes[:combo_limit], 2))
        random.shuffle(all_pairs)
        
        max_checks = min(500, max(60, self.ctx.num_pipes))
        
        for d1, d2 in all_pairs[:max_checks]:
            if indices_copy[d1] == 0 or indices_copy[d2] == 0: continue
            
            test_sol = list(indices_copy)
            test_sol[d1] -= 1
            test_sol[d2] -= 1
            locked = {d1, d2}
            
            healed_sol, is_feasible, boosts = self.ls.heal_network(test_sol, locked)
            if is_feasible:
                h_c, h_p, _, _ = self.ctx.get_cached_stats(healed_sol) 
                h_surplus = h_p - self.ctx.simulator.config.h_min
                h_score = h_c - (h_surplus * dyn_bonus)
                candidates.append((h_score, healed_sol, locked, boosts))
                
        if not candidates: return None, None, ""
        candidates.sort(key=lambda x: x[0])
        validated_candidates = []
        
        for _, healed_sol, locked, boosts in candidates[:5]:
            squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
            sq_c, _, _, _ = self.ctx.get_cached_stats(squeezed)
            validated_candidates.append((sq_c, healed_sol, locked, boosts))
                
        if not validated_candidates: return None, None, ""
        validated_candidates.sort(key=lambda x: x[0])
        chosen = random.choice(validated_candidates[:3])
        _, best_final_sol, best_locked, best_boosts = chosen
        
        return best_final_sol, best_locked, f"SYNC-TRIM: Shrunk Pipes {[p+1 for p in best_locked]}. Healed {best_boosts}x."

    def loop_balancing_kick(self, indices, dyn_bonus):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        try: cycles = nx.cycle_basis(self.ctx.base_G_flow)
        except: return None, None, ""
        if not cycles: return None, None, "No cycles found"

        best_drop_achieved = -1
        candidates = []

        for cycle_nodes in cycles:
            cycle_indices = []
            full_cycle = cycle_nodes + [cycle_nodes[0]]
            for u, v in zip(full_cycle[:-1], full_cycle[1:]):
                if (u, v) in self.ctx.edge_to_pipe: cycle_indices.append(self.ctx.edge_to_pipe[(u, v)])

            for candidate_idx in cycle_indices:
                curr_d_idx = indices[candidate_idx]
                if curr_d_idx < 2: continue 
                
                max_drop = min(3, curr_d_idx) 
                for drop in range(max_drop, 0, -1):
                    if drop < best_drop_achieved: continue
                        
                    kicked, locked = list(indices), set()
                    kicked[candidate_idx] -= drop
                    locked.add(candidate_idx)
                    
                    healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
                    if is_feasible:
                        test_squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=4, quick_mode=True, dyn_bonus=dyn_bonus)
                        sq_cost, _, _, _ = self.ctx.get_cached_stats(test_squeezed)
                        msg = f"FLOW STEER: Cut Pipe {candidate_idx + 1} (-{drop}). Healed {boosts}x."
                        candidates.append((sq_cost, healed_sol, locked, msg))
                        best_drop_achieved = max(best_drop_achieved, drop)

        if not candidates: return None, None, "FLOW STEER: Exhaustive search found no valid bypass."
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])
        return chosen[1], chosen[2], chosen[3]
    
    def diameter_diversity_kick(self, indices, stagnation_counter):
        # 🔴 ПАТЧ: Стратегія спрацьовує завжди, коли її викликає UCB1
        kicked, locked = list(indices), set()
        
        base_n = max(5, int(self.ctx.num_pipes * 0.12))
        aggression_step = max(0.5, self.ctx.num_pipes / 100.0) 
        n_to_change = min(self.ctx.num_pipes // 2, base_n + int(stagnation_counter * aggression_step))
        
        pipes_to_change = random.sample(range(self.ctx.num_pipes), n_to_change)
        
        # Визначаємо найпопулярніший діаметр для внесення хаосу
        from collections import Counter
        diam_counts = Counter(indices)
        most_common_diam = diam_counts.most_common(1)[0][0]
        
        for p in pipes_to_change:
            current_d = kicked[p]
            if current_d == most_common_diam:
                if random.random() < 0.5 and current_d < self.ctx.max_d_idx:
                    kicked[p] += 1
                elif current_d > 0:
                    kicked[p] -= 1
            else:
                kicked[p] = most_common_diam
            locked.add(p)
            
        healed_sol, is_feas, _ = self.ls.heal_network(kicked, locked)
        if not is_feas: return None, None, ""
        
        return healed_sol, locked, f"DIAM-DIVERSITY: Changed {len(locked)} pipes to break homogeneity."
    
    def ruin_and_recreate_kick(self, indices, stagnation_counter):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        # 🔴 ПАТЧ: Ротація точки руйнування при глибокій стагнації
        if stagnation_counter > 8:
            all_nodes = list(self.ctx.base_G_flow.nodes())
            ruin_center = random.choice(all_nodes)
        else:
            ruin_center = crit_node

        max_pct = 0.20 if self.ctx.num_pipes <= 50 else 0.40
        base_pct = random.uniform(0.10, max_pct / 2)
        target_pct = min(max_pct, base_pct + (stagnation_counter * 0.025)) 
        target_pipes = max(3, int(self.ctx.num_pipes * target_pct))
        
        pipe_with_dist = []
        seen_pipes = set()
        
        try:
            # Шукаємо від ruin_center замість crit_node
            for node, dist in nx.single_source_shortest_path_length(self.ctx.base_G_flow, ruin_center, cutoff=15).items():
                for p in self.ctx.node_to_pipes.get(node, []):
                    if p not in seen_pipes:
                        seen_pipes.add(p)
                        pipe_with_dist.append((dist, p))
        except:
            pass

        if not pipe_with_dist: return None, None, ""
        
        pipe_with_dist.sort(key=lambda x: x[0])
        cluster_pipes = set(p for _, p in pipe_with_dist[:target_pipes])

        kicked = list(indices)
        ruined_count = 0
        
        for p in cluster_pipes:
            if kicked[p] > 0:
                kicked[p] = 0
                ruined_count += 1

        if ruined_count == 0: return None, None, ""

        healed_sol, is_feas, boosts = self.ls.heal_network(kicked, set())
        if not is_feas: return None, None, ""

        actual_radius = max((d for d, _ in pipe_with_dist[:target_pipes]), default=0)
        
        return healed_sol, cluster_pipes, f"R&R (LNS): Ruined {ruined_count} pipes (~{target_pct:.0%} net, Rad: {actual_radius}). Rebuilt {boosts}x."

    def ils_perturbation_kick(self, indices, stagnation_counter):
        """ILS: Великий сліпий стрибок для виходу з макро-долини"""
        max_pct = 0.15 if self.ctx.num_pipes <= 50 else 0.35
        pct = min(max_pct, (max_pct / 2) + stagnation_counter * 0.02)
        n_perturb = max(4, int(self.ctx.num_pipes * pct))
        
        perturbed = list(indices)
        pipes = random.sample(range(self.ctx.num_pipes), n_perturb)
        
        for p in pipes:
            perturbed[p] = random.randint(0, self.ctx.max_d_idx)
            
        healed, ok, _ = self.ls.heal_network(perturbed, set())
        if not ok: return None, None, ""
        
        msg = f"ILS-PERTURB: Random jump on {n_perturb} pipes ({pct:.0%}), new basin entry."
        return healed, set(pipes), msg

    # 🔴 ВИПРАВЛЕНО: Додано аргумент dyn_bonus
    def segment_restart_kick(self, indices, dyn_bonus):
        """Segment Restart: Заморожує Топ-20% і скидає решту 80% до максимуму"""
        n = self.ctx.num_pipes
        freeze_pct = 0.50 if n <= 50 else 0.20
        n_freeze = max(5, int(n * freeze_pct)) 
        
        frozen_pipes = set(self.ls.get_high_impact_pipes(indices, n_freeze))
        
        fresh = list(indices)
        for p in range(n):
            if p not in frozen_pipes:
                fresh[p] = self.ctx.max_d_idx 
                
        # 🔴 ВИПРАВЛЕНО: Використовуємо актуальний бонус для правильного Squeeze
        squeezed = self.ls.gradient_squeeze(fresh, locked_pipes=frozen_pipes, max_passes=None, quick_mode=False, dyn_bonus=dyn_bonus)
        
        cost, p_val, feas, _ = self.ctx.get_cached_stats(squeezed)
        if not feas or p_val < self.ctx.simulator.config.h_min:
            return None, None, ""
            
        return squeezed, frozen_pipes, f"SEGMENT-RESTART: Froze {n_freeze} pipes, Top-Down squeeze on {n-n_freeze}."

    def crossover_with_peer_kick(self, my_sol, peer_sol, my_cost, peer_cost):
        """Схрещування з успішним рішенням іншого воркера"""
        child = []
        total = my_cost + peer_cost
        p_mine = peer_cost / total 
        
        locked = set()
        for i, (a, b) in enumerate(zip(my_sol, peer_sol)):
            if a == b:
                child.append(a)
            else:
                chosen = a if random.random() < p_mine else b
                child.append(chosen)
                locked.add(i) # Блокуємо змінені труби
                
        # 🔴 ПАТЧ 3 (ВИПРАВЛЕНО): Передаємо locked, щоб heal_network не затер наші свіжі зміни!
        healed, ok, _ = self.ls.heal_network(child, locked)
        
        return healed, locked, f"IPC-CROSSOVER: Merged with peer (Mine: {p_mine:.0%}, Peer: {1-p_mine:.0%})."
    
    def corridor_search_kick(self, sol_A, sol_B):
        """Досліджує простір між двома рішеннями (Hamming 30-70)"""
        child = list(sol_A)
        diff_pipes = [i for i in range(self.ctx.num_pipes) if sol_A[i] != sol_B[i]]
        
        if not (30 <= len(diff_pipes) <= 70):
            return None, None, ""
            
        locked = set()
        for i in diff_pipes:
            # 🔴 Зберігаємо фізичну структуру: беремо ген від А або Б, а не середнє
            if random.random() < 0.5:
                child[i] = sol_B[i]
            locked.add(i)
            
        healed, ok, boosts = self.ls.heal_network(child, locked)
        if not ok: return None, None, ""
        
        return healed, locked, f"CORRIDOR: Interpolated {len(diff_pipes)} diff pipes. Healed {boosts}x."

# =============================================================================
# 5. ORCHESTRATOR (Parallel Island Model Architecture)
# =============================================================================
class AnalyticalSolver:
    worker_task = None 
    
    def __init__(self, simulator_instance, available_diameters, v_opt=1.0, max_sims=None, 
                 time_limit_sec=None, pool=None, log_dir=None, n_workers=1):
        
        self.ctx = SolverContext(simulator_instance, available_diameters, v_opt)
        self.ls = LocalSearch(self.ctx)
        self.kicker = KickStrategies(self.ctx, self.ls)
        
        self.mp_pool = pool
        self.pool = SolutionPool(self.ctx)
        
        self.log_dir = log_dir
        self.n_workers = n_workers
        
        n = self.ctx.num_pipes
        if n < 50: self.network_class = "SMALL"
        elif n < 200: self.network_class = "MEDIUM"
        elif n < 1000: self.network_class = "LARGE"
        else: self.network_class = "XLARGE"
            
        self._configure_for_scale()
        
        if max_sims is not None:
            self.max_sims = max_sims
        else:
            self.max_sims = self.BASE_SIM_BUDGET
            
        if time_limit_sec is not None:
            self.time_limit_sec = time_limit_sec
        else:
            secs_needed = self.max_sims / max(self.ctx.sim_speed, 1.0)
            self.time_limit_sec = max(3600, secs_needed * 2.0)
            
        print(f"[Config] Network: {self.network_class} ({self.ctx.num_pipes} pipes)")
        print(f"  Target Budget: {self.max_sims:,} sims (≈{self.time_limit_sec/60:.1f} min)")
        print(f"  BeamWidth={self.BEAM_WIDTH}")
        
        self.strategies = ["TOPO-DIV", "LOOP_BALANCE", "SYNC_TRIM", "FINISHER", "BOTTLENECK", "SHOCK"]
        if nx.is_tree(self.ctx.base_G_flow):
            self.strategies.remove("LOOP_BALANCE")
        self.strategies.append("DIAM_DIVERSITY")
        self.strategies.append("RUIN_RECREATE")
        
        self.strat_wins = {s: 1.0 for s in self.strategies}
        self.strat_tries = {s: 1 for s in self.strategies}

    def _configure_for_scale(self):
        n = self.ctx.num_pipes
        cls = self.network_class
        
        self.BEAM_WIDTH = {
            "SMALL": max(5, n//10), 
            "MEDIUM": max(5, n//20), 
            "LARGE": 8, 
            "XLARGE": 6
        }[cls]
        
        self.SINGLE_CANDIDATES = {"SMALL": max(6, n//5), "MEDIUM": max(6, n//10), "LARGE": max(6, n//20), "XLARGE": 10}[cls]
        
        self.BASE_SIM_BUDGET = {
            "SMALL": 30000, 
            "MEDIUM": 100000, 
            "LARGE": 1000000, 
            "XLARGE": 3000000
        }[cls]

    def _calculate_ideal_d(self, flow, target_v):
        if abs(flow) < 1e-6: return 0
        d_ideal = math.sqrt((4.0 * abs(flow)) / (math.pi * target_v))
        pos = bisect.bisect_left(self.ctx.diameters, d_ideal)
        if pos < len(self.ctx.diameters): return pos
        return self.ctx.max_d_idx

    def _make_diverse_seeds(self):
        seeds = []
        for v in [1.0, 1.2, 0.8]:
            idx_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            for _ in range(10):
                real_diams = [self.ctx.diameters[i] for i in idx_sol]
                flows, _ = self.ctx.simulator.get_hydraulic_state(real_diams)
                
                new_idx = [self._calculate_ideal_d(q, v) for q in flows] 
                if new_idx == idx_sol: break
                idx_sol = new_idx
                
            _, p, feas, _ = self.ctx.get_cached_stats(idx_sol)
            if not feas or p < self.ctx.simulator.config.h_min:
                idx_sol, is_healed, _ = self.ls.heal_network(idx_sol, set())
                if not is_healed: continue 
            squeezed_sol = self.ls.gradient_squeeze(idx_sol, quick_mode=True)
            seeds.append(squeezed_sol)
        return seeds
    
    def _make_relinked_seeds(self, archive):
        import random
        import itertools
        seeds = []
        
        if len(archive) < 2:
            return []

        top_sols = [x[1] for x in archive[:4]]
        pairs = list(itertools.combinations(top_sols, 2))
        random.shuffle(pairs)

        max_steps = {"SMALL": 999, "MEDIUM": 30, "LARGE": 15, "XLARGE": 10}[self.network_class]

        for sol_A, sol_B in pairs:
            diff_indices = [i for i in range(self.ctx.num_pipes) if sol_A[i] != sol_B[i]]
            current_path_sol = list(sol_A)
            
            steps_taken = 0
            while diff_indices and len(seeds) < self.BEAM_WIDTH and steps_taken < max_steps:
                steps_taken += 1
                candidates = diff_indices if len(diff_indices) <= 10 else random.sample(diff_indices, 10)
                
                best_idx, best_score = -1, float('inf')
                best_feas, best_p = False, 0
                
                for idx in candidates:
                    test_sol = list(current_path_sol)
                    test_sol[idx] = sol_B[idx]
                    c, p, feas, _ = self.ctx.get_cached_stats(test_sol)
                    
                    score = c if (feas and p >= self.ctx.simulator.config.h_min) else c * 1.5
                    
                    if score < best_score:
                        best_score, best_idx = score, idx
                        best_feas, best_p = feas, p
                        
                current_path_sol[best_idx] = sol_B[best_idx]
                diff_indices.remove(best_idx)
                
                if best_feas and best_p >= self.ctx.simulator.config.h_min:
                    squeezed = self.ls.gradient_squeeze(current_path_sol, quick_mode=True)
                    seeds.append(squeezed)
                    
            if len(seeds) >= self.BEAM_WIDTH:
                break
                
        while len(seeds) < self.BEAM_WIDTH:
            seeds.append(list(top_sols[0]))
            
        return seeds

    def _make_warm_seeds(self, archive, worker_id=0):
        import random
        
        if not archive:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🏴‍☠️ ADVENTURER (Empty Archive Fallback)")
            return self._make_diverse_seeds()
        
        seeds = []
        
        if worker_id in [3, 7, 11, 15, 19]: 
            role = "ADVENTURER"
        elif worker_id == 0: role = "EXPLOITER"
        elif worker_id == 1: role = "RELINKER"
        elif worker_id == 2: role = "EXPLORER"
        else:
            caste_roll = random.random()
            if caste_roll < 0.30: role = "EXPLOITER"
            elif caste_roll < 0.60: role = "RELINKER"
            elif caste_roll < 0.85: role = "EXPLORER"
            else: role = "ADVENTURER"
            
        if role == "RELINKER" and len(archive) >= 2:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🧬 RELINKER (Path-Relinking between Top-4)")
            return self._make_relinked_seeds(archive)
            
        elif role == "EXPLOITER" or (role == "RELINKER" and len(archive) < 2):
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🎯 EXPLOITER (Micro-mutations on Top-2)")
            top_sols = [x[1] for x in archive[:2]]
            n_perturb = max(1, self.ctx.num_pipes // 15) 
            deltas = [-1, 1]                             
            
        elif role == "EXPLORER":
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🔭 EXPLORER (Macro-mutations on Top-5)")
            top_sols = [x[1] for x in archive]
            n_perturb = max(3, self.ctx.num_pipes // 8)  
            deltas = [-2, -1, 1, 2]                      
            
        else:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🏴‍☠️ ADVENTURER (Cold Start - Ignored Archive)")
            return self._make_diverse_seeds()
            
        for _ in range(self.BEAM_WIDTH):
            base_sol = random.choice(top_sols)
            perturbed = list(base_sol)
            pipes_to_shake = random.sample(range(self.ctx.num_pipes), n_perturb)
            
            for p in pipes_to_shake:
                new_idx = max(0, min(self.ctx.max_d_idx, perturbed[p] + random.choice(deltas)))
                perturbed[p] = new_idx
                
            healed_sol, is_feas, _ = self.ls.heal_network(perturbed, set())
            if is_feas:
                squeezed = self.ls.gradient_squeeze(healed_sol, quick_mode=True)
                seeds.append(squeezed)
            else:
                seeds.append(base_sol) 
            
        return seeds
    
    def _make_reserve_pool(self, size=4):
        reserve = []
        for _ in range(size):
            # Рандомна топологія
            seed = [random.randint(0, self.ctx.max_d_idx) for _ in range(self.ctx.num_pipes)]
            healed, ok, _ = self.ls.heal_network(seed, set())
            if ok:
                # 🔴 ЗАХИСТ ВІД ЛІЙКИ: Заморожуємо 25% труб під час спуску
                n_freeze = self.ctx.num_pipes // 4
                frozen = set(random.sample(range(self.ctx.num_pipes), n_freeze))
                squeezed = self.ls.gradient_squeeze(healed, locked_pipes=frozen, quick_mode=True)
                reserve.append(squeezed)
        return reserve

    def _run_single_search(self, seeds, time_budget, global_best_cost, worker_id=0, shared_progress=None):
        if not seeds:
            self.ctx.log("[WARNING] Seed generation returned empty. Using safe fallback.")
            seeds = [[self.ctx.max_d_idx] * self.ctx.num_pipes]
            
        run_best_cost = float('inf')
        run_best_sol = None
        
        start_time = time.time()
        self.pool.clear_all()
        
        # 🔴 ГЕНЕРУЄМО РЕЗЕРВ (Diversity Control)
        reserve_pool = self._make_reserve_pool(size=4)
        
        best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds])
        base_dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001 
        
        valid_sols = []
        for s in seeds:
            c, p, feas, _ = self.ctx.get_cached_stats(s)
            p_surplus = p - self.ctx.simulator.config.h_min
            score = c - (p_surplus * base_dyn_bonus)
            self.pool.active_pool.append((score, c, s))
            self.pool.add_to_tabu(s, c)
            if feas and p >= self.ctx.simulator.config.h_min:
                valid_sols.append((c, s))
                
        if valid_sols:
            run_best_cost = min(x[0] for x in valid_sols)
            run_best_sol = next(x[1] for x in valid_sols if x[0] == run_best_cost)
        else:
            run_best_cost = global_best_cost
            run_best_sol = min(self.pool.active_pool, key=lambda x: x[0])[2] 

        stagnation_counter = 0
        self.ctx.log(f"   > Seeds initialized. Baseline Target: {run_best_cost/1e6:.4f}M$")

        round_idx = 0
        last_published_cost = float('inf')
        
        while True:
            # 🔴 СТОХАСТИЧНЕ ЗВАЖУВАННЯ (Jitter): Змінюємо кут ландшафту щораунду
            jitter_factor = random.uniform(0.70, 1.30)
            base_dyn_bonus = min(run_best_cost, global_best_cost) * 0.001 * jitter_factor

            # IPC Читання
            if shared_progress is not None:
                gb = shared_progress.get('global_best')
                if gb and gb[0] < global_best_cost:
                    global_best_cost = gb[0]
                    self.ctx.log(f"   > [IPC] 📡 Received new Global Bound from peer: {global_best_cost/1e6:.4f}M$")

            if shared_progress is not None and worker_id is not None:
                shared_progress[worker_id] = {
                    "round": round_idx + 1,
                    "sims": self.ctx.sim_count,
                    "best_cost": run_best_cost
                }
            
            elapsed = time.time() - start_time
            if elapsed > time_budget or self.ctx.sim_count >= self.max_sims: 
                break
                
            just_flushed = False
            progress_ratio = min(1.0, self.ctx.sim_count / max(1, self.max_sims))
            is_late_game = progress_ratio > 0.5 
            stag_limit = 4 + int(4 * progress_ratio) 
            
            if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()
            next_gen = []
            
            if round_idx % 3 == 0 or stagnation_counter == 0:
                swapped = self.ls.swap_search(run_best_sol, base_dyn_bonus)
                c, p, feas, _ = self.ctx.get_cached_stats(swapped)
                if feas and p >= self.ctx.simulator.config.h_min and c < run_best_cost:
                    is_ghost = False
                    if (run_best_cost - c) > (run_best_cost * 0.02):
                        is_ghost = self.ctx.is_ghost_solution(swapped, c)
                        
                    if is_ghost:
                        self.ctx.log(f"   > [SHIELD] Swap Heuristic illusion blocked ({c/1e6:.4f}M$)!")
                    else:
                         diff = run_best_cost - c
                         run_best_cost, run_best_sol = c, swapped
                         self.ctx.log(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$)")
                         
                         if run_best_cost < last_published_cost:
                             if shared_progress is not None:
                                 shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                             last_published_cost = run_best_cost

                         if run_best_cost < global_best_cost:
                             global_best_cost = run_best_cost
                             if shared_progress is not None:
                                 current_gb = shared_progress.get('global_best', (float('inf'), []))
                                 if run_best_cost < current_gb[0]:
                                     shared_progress['global_best'] = (run_best_cost, list(run_best_sol))

                         stagnation_counter = 0
                         self.pool.kick_tabu_set.clear()
                         
                         # Адаптивний бонус для оцінки
                         p_surplus = p - self.ctx.simulator.config.h_min
                         eff_bonus = base_dyn_bonus * 0.3 if p_surplus < 1.0 else (base_dyn_bonus * 2.0 if p_surplus > 15.0 else base_dyn_bonus)
                         
                         self.pool.active_pool.insert(0, (c - (p_surplus * eff_bonus) - (run_best_cost * 0.1), c, swapped))

            if stagnation_counter >= 1 or round_idx % 2 == 0:
                n_total = sum(self.strat_tries.values())
                sims_before_kick = self.ctx.sim_count
                
                peer_archive = []
                if shared_progress is not None:
                    if run_best_cost < last_published_cost:
                        shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                        last_published_cost = run_best_cost
                        
                    for i in range(self.n_workers):
                        if i != worker_id:
                            peer_data = shared_progress.get(f'best_sol_{i}')
                            if peer_data: peer_archive.append(peer_data)

                peer_to_cross = None
                if stagnation_counter >= stag_limit * 3:
                    strategy = "SEGMENT_RESTART"
                    self.pool.add_basin_to_tabu(run_best_sol) 
                elif stagnation_counter >= stag_limit * 2 and peer_archive:
                    # 🔴 CORRIDOR SEARCH (Шукаємо сусідів на дистанції 30-70)
                    corridor_peers = [p for p in peer_archive if 30 <= self.pool.hamming_distance(run_best_sol, p[1]) <= 70]
                    diverse_peers = [p for p in peer_archive if self.pool.hamming_distance(run_best_sol, p[1]) > 70]
                    
                    if corridor_peers:
                        strategy = "CORRIDOR_SEARCH"
                        peer_to_cross = random.choice(corridor_peers)
                    elif diverse_peers:
                        strategy = "IPC_CROSSOVER"
                        peer_to_cross = min(diverse_peers, key=lambda x: x[0])
                    else:
                        strategy = "ILS_PERTURBATION"
                elif stagnation_counter >= stag_limit * 1.5:
                    strategy = "ILS_PERTURBATION"
                elif stagnation_counter > 0 and stagnation_counter % 12 == 0:
                    strategy = "RUIN_RECREATE"
                else:
                    valid_strats = [s for s in self.strategies if s not in ["SEGMENT_RESTART", "ILS_PERTURBATION", "IPC_CROSSOVER", "CORRIDOR_SEARCH"]]
                    strategy = max(valid_strats, key=lambda s: (self.strat_wins[s] / self.strat_tries.get(s, 1)) + math.sqrt(2 * math.log(max(1, n_total)) / self.strat_tries.get(s, 1)))
                
                self.strat_tries[strategy] = self.strat_tries.get(strategy, 0) + 1
                self.ctx.log(f"[FORCE] Escalation Level: {stagnation_counter}/{stag_limit} -> Applying '{strategy}'...")
                
                kick_mode = round_idx % 3
                if kick_mode == 0: kick_target = run_best_sol
                elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
                else:
                    pool_sols = [x[2] for x in self.pool.active_pool[:3]]
                    kick_target = random.choice(pool_sols) if pool_sols else run_best_sol
                
                forced_sol, locked, path_sig = None, None, None
                
                if strategy == "SEGMENT_RESTART": forced_sol, locked, log_msg = self.kicker.segment_restart_kick(kick_target, base_dyn_bonus)
                elif strategy == "IPC_CROSSOVER": forced_sol, locked, log_msg = self.kicker.crossover_with_peer_kick(kick_target, peer_to_cross[1], run_best_cost, peer_to_cross[0])
                elif strategy == "CORRIDOR_SEARCH": forced_sol, locked, log_msg = self.kicker.corridor_search_kick(kick_target, peer_to_cross[1])
                elif strategy == "ILS_PERTURBATION": forced_sol, locked, log_msg = self.kicker.ils_perturbation_kick(kick_target, stagnation_counter)
                elif strategy == "SHOCK": forced_sol, locked, log_msg = self.kicker.forcing_hand_kick(kick_target)
                elif strategy == "BOTTLENECK": forced_sol, locked, log_msg = self.kicker.upstream_bottleneck_kick(kick_target)
                elif strategy == "LOOP_BALANCE": forced_sol, locked, log_msg = self.kicker.loop_balancing_kick(kick_target, base_dyn_bonus)
                elif strategy == "DIAM_DIVERSITY": forced_sol, locked, log_msg = self.kicker.diameter_diversity_kick(kick_target, stagnation_counter) 
                elif strategy == "RUIN_RECREATE": forced_sol, locked, log_msg = self.kicker.ruin_and_recreate_kick(kick_target, stagnation_counter) 
                elif strategy == "FINISHER": forced_sol, locked, log_msg = self.kicker.micro_trim_kick(kick_target)
                elif strategy == "SYNC_TRIM": forced_sol, locked, log_msg = self.kicker.sync_trim_kick(kick_target, base_dyn_bonus)
                else: forced_sol, locked, log_msg, path_sig = self.kicker.topological_inversion_kick(kick_target, self.pool.kick_tabu_set)
                
                if forced_sol and locked:
                    self.ctx.log(f"     -> {log_msg}")
                    
                    if strategy == "SEGMENT_RESTART":
                        final_sol = forced_sol 
                    elif strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "CORRIDOR_SEARCH"]:
                        final_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=set(), max_passes=None, quick_mode=False, dyn_bonus=base_dyn_bonus)
                    elif strategy == "RUIN_RECREATE":
                        final_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=4, quick_mode=True, dyn_bonus=base_dyn_bonus)
                    else:
                        starved_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=8, dyn_bonus=base_dyn_bonus)
                        final_sol = self.ls.gradient_squeeze(starved_sol, dyn_bonus=base_dyn_bonus)
                    
                    c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                    
                    if feas and p >= self.ctx.simulator.config.h_min:
                        p_surplus = p - self.ctx.simulator.config.h_min
                        
                        # 🔴 АДАПТИВНИЙ БОНУС ДЛЯ ОЦІНКИ (Pareto-logic)
                        eff_bonus = base_dyn_bonus * 0.3 if p_surplus < 1.0 else (base_dyn_bonus * 2.0 if p_surplus > 15.0 else base_dyn_bonus)
                        score = c - (p_surplus * eff_bonus)
                        
                        water_level = global_best_cost * (1.05 - 0.05 * progress_ratio)
                        
                        if c < run_best_cost:
                            is_ghost = False
                            if (run_best_cost - c) > (run_best_cost * 0.02):
                                is_ghost = self.ctx.is_ghost_solution(final_sol, c)
                                
                            if is_ghost:
                                self.ctx.log(f"   > [SHIELD] Force Heuristic illusion blocked ({c/1e6:.4f}M$)!")
                            else:
                                diff = run_best_cost - c
                                run_best_cost, run_best_sol = c, final_sol
                                self.pool.active_pool.insert(0, (score, c, final_sol))
                                
                                stagnation_counter = 0 
                                
                                if strategy in self.strategies: self.strat_wins[strategy] += 3.0 
                                self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$)")
                                
                                if run_best_cost < last_published_cost:
                                    if shared_progress is not None:
                                        shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                                    last_published_cost = run_best_cost

                                if run_best_cost < global_best_cost:
                                    global_best_cost = run_best_cost
                                    if shared_progress is not None:
                                        current_gb = shared_progress.get('global_best', (float('inf'), []))
                                        if run_best_cost < current_gb[0]:
                                            shared_progress['global_best'] = (run_best_cost, list(run_best_sol))

                        elif c < water_level and not self.pool.is_basin_tabu(final_sol):
                            self.pool.active_pool.append((score * 1.05, c, final_sol))
                            
                            if strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "SEGMENT_RESTART", "CORRIDOR_SEARCH"]:
                                stagnation_counter = 0 
                                
                            if strategy in self.strategies: self.strat_wins[strategy] += 0.5
                            self.ctx.log(f"     -> Added to Pool (Water Level Accept): {c/1e6:.4f}M$")
                        else:
                            self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {c/1e6:.4f}M$")
                            
                        if path_sig: self.pool.kick_tabu_set.add(path_sig)
                    else:
                         self.ctx.log(f"     -> Injection Failed: Infeasible (P={p:.2f}m)")

            for _, _, parent_sol in self.pool.active_pool:
                unit_losses = self.ctx.get_cached_heuristics(parent_sol)
                high_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i], reverse=True)
                low_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i]) 
                
                if self.network_class in ("LARGE", "XLARGE"):
                    top_k = max(20, self.ctx.num_pipes // 5)
                    focus_pipes = set(self.ls.get_high_impact_pipes(parent_sol, top_k))
                    high_friction = [p for p in high_friction if p in focus_pipes]
                    low_friction = [p for p in low_friction if p in focus_pipes]
                
                for pipe_idx in high_friction[:self.SINGLE_CANDIDATES]:
                    s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "upgrade", base_dyn_bonus)
                    if sol: next_gen.append((s, c, sol))
                
                for pipe_idx in low_friction[:8]:
                     s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "downgrade", base_dyn_bonus)
                     if sol: next_gen.append((s, c, sol))
                     
                if len(high_friction) >= 2:
                    for p1, p2 in itertools.combinations(high_friction[:5], 2): 
                        s, c, sol = self.ls.evaluate_candidate(parent_sol, [p1, p2], "upgrade", base_dyn_bonus)
                        if sol: next_gen.append((s, c, sol))
            
            if not next_gen:
                if not just_flushed: stagnation_counter += 1
                round_idx += 1
                
                if not self.pool.active_pool:
                    self.ctx.log("     [EMERGENCY] Pool empty! Forcing ILS respawn to break loop...")
                    healed, ok, _ = self.kicker.ils_perturbation_kick(run_best_sol, 15)
                    if ok:
                        c, p, feas, _ = self.ctx.get_cached_stats(healed)
                        if feas and p >= self.ctx.simulator.config.h_min:
                            self.pool.active_pool.append((c, c, healed))
                            stagnation_counter = 0 
                
                self.pool.current_round = round_idx
                continue

            next_gen.sort(key=lambda x: x[0]) 
            unique_next_pool = []
            found_new_record = False
            
            max_d = max(2, self.ctx.num_pipes // 7) 
            min_dist = max(1, int(max_d * (1.0 - (progress_ratio ** 2))))
            
            for rank, (score, cost, sol) in enumerate(next_gen):
                if self.pool.is_tabu(sol, cost): continue
                if len(unique_next_pool) < self.BEAM_WIDTH:
                    if rank == 0: refined_sol = self.ls.gradient_squeeze(sol, dyn_bonus=base_dyn_bonus)
                    elif rank < 3: refined_sol = self.ls.gradient_squeeze(sol, max_passes=4, quick_mode=not is_late_game, dyn_bonus=base_dyn_bonus)
                    else: refined_sol = sol 

                    is_diverse = True
                    for _, _, peer_sol in unique_next_pool:
                        if self.pool.hamming_distance(refined_sol, peer_sol) < min_dist:
                            is_diverse = False; break
                            
                    if not is_diverse and not just_flushed: continue

                    real_cost, p_min, feas_min, _ = self.ctx.get_cached_stats(refined_sol)
                    if feas_min and p_min >= self.ctx.simulator.config.h_min:
                        real_p_surplus = p_min - self.ctx.simulator.config.h_min
                        
                        # 🔴 Адаптивний бонус для Beam Search оцінки
                        eff_bonus = base_dyn_bonus * 0.3 if real_p_surplus < 1.0 else (base_dyn_bonus * 2.0 if real_p_surplus > 15.0 else base_dyn_bonus)
                        
                        unique_next_pool.append((real_cost - (real_p_surplus * eff_bonus), real_cost, refined_sol))
                        self.pool.add_to_tabu(refined_sol, real_cost)
                        
                        if real_cost < run_best_cost:
                            is_ghost = False
                            if (run_best_cost - real_cost) > (run_best_cost * 0.02):
                                is_ghost = self.ctx.is_ghost_solution(refined_sol, real_cost)
                                
                            if is_ghost:
                                self.ctx.log(f"   > [SHIELD] Beam Heuristic illusion blocked ({real_cost/1e6:.4f}M$). Discarding!")
                                continue 

                            diff = run_best_cost - real_cost
                            run_best_cost, run_best_sol = real_cost, refined_sol
                            found_new_record = True
                            self.ctx.log(f"   > [R{round_idx+1}] 💎 New Record: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$)")
                            
                            if run_best_cost < last_published_cost:
                                if shared_progress is not None:
                                    shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                                last_published_cost = run_best_cost

                            if run_best_cost < global_best_cost:
                                global_best_cost = run_best_cost
                                if shared_progress is not None:
                                    current_gb = shared_progress.get('global_best', (float('inf'), []))
                                    if run_best_cost < current_gb[0]:
                                        shared_progress['global_best'] = (run_best_cost, list(run_best_sol))

            if found_new_record or just_flushed:
                stagnation_counter = 0; self.pool.kick_tabu_set.clear()
            else:
                stagnation_counter += 1
                if stagnation_counter % 8 == 0: self.pool.kick_tabu_set.clear()

            if unique_next_pool:
                unique_next_pool.sort(key=lambda x: x[0])
                self.pool.active_pool = unique_next_pool[:self.BEAM_WIDTH]
                
                # 🔴 DIVERSITY CONTROL: Ін'єкція "свіжої крові" при гомогенізації пулу
                if len(self.pool.active_pool) >= 3:
                    sols = [x[2] for x in self.pool.active_pool]
                    pairs = list(itertools.combinations(range(len(sols)), 2))
                    avg_dist = sum(self.pool.hamming_distance(sols[a], sols[b]) for a, b in pairs) / len(pairs)
                    
                    diversity_threshold = self.ctx.num_pipes // 8
                    if avg_dist < diversity_threshold and reserve_pool:
                        self.ctx.log(f"   [DIVERSITY] Pool homogenizing (dist={avg_dist:.1f}). Injecting reserve seed!")
                        fresh = reserve_pool.pop(0)  # Беремо з резерву
                        c, p, feas, _ = self.ctx.get_cached_stats(fresh)
                        if feas:
                            p_surplus = p - self.ctx.simulator.config.h_min
                            score = c - (p_surplus * base_dyn_bonus)
                            self.pool.active_pool[-1] = (score, c, fresh)  # Замінюємо найгірший слот
                
            self.pool.current_round = round_idx
            round_idx += 1
                
        if shared_progress is not None and worker_id is not None:
            shared_progress[worker_id] = {"round": "DONE", "sims": self.ctx.sim_count, "best_cost": run_best_cost}

        return run_best_cost, run_best_sol

    def solve_standalone(self):
        print("\n[AnalyticalSolver] ⚡ Initiating Island Model Search...\n")
        start_time = time.time()
        global_best_cost = float('inf')
        global_best_sol = None
        global_archive = []

        epochs = {"SMALL": 2, "MEDIUM": 2, "LARGE": 4, "XLARGE": 4}[self.network_class]
        time_per_epoch = self.time_limit_sec / epochs

        if self.mp_pool:
            manager = multiprocessing.Manager()
            shared_progress = manager.dict()
            for i in range(self.n_workers):
                shared_progress[i] = 0
        else:
            shared_progress = None

        for epoch in range(epochs):
            mode_str = "PARALLEL" if self.mp_pool else "SEQUENTIAL"
            print("="*46)
            print(f" [EPOCH {epoch+1}/{epochs}] {mode_str} Workers: {self.n_workers} | Time Limit: {time_per_epoch/60:.1f} min")
            print("="*46)
            
            seed_modifier = random.randint(1, 10000)
            tasks = []
            for i in range(self.n_workers):
                tasks.append((
                    self.ctx.diameters, self.ctx.v_opt, time_per_epoch, 
                    global_best_cost, global_archive, 
                    seed_modifier + i, i, shared_progress, self.log_dir, epoch
                ))

            epoch_results = []
            
            # --- ГІЛКА МУЛЬТИПРОЦЕСИНГУ ---
            if self.mp_pool:
                async_results = []
                for t in tasks:
                    res = self.mp_pool.apply_async(self.worker_task, (t,))
                    async_results.append((t[6], res))

                epoch_start_time = time.time()
                last_print_time = 0
                print_interval = {"SMALL": 30.0, "MEDIUM": 60.0, "LARGE": 180.0, "XLARGE": 300.0}[self.network_class]
                
                while True:
                    all_done = all(res.ready() for _, res in async_results)
                    if all_done: break
                        
                    curr_time = time.time()
                    if curr_time - last_print_time >= print_interval:
                        last_print_time = curr_time
                        elapsed_total = curr_time - start_time # Рахуємо від початку запуску!
                        m, s = divmod(int(elapsed_total), 60)
                        
                        status_parts = []
                        total_sims = 0
                        live_best = global_best_cost
                        
                        for wid in range(self.n_workers):
                            prog = shared_progress.get(wid, 0)
                            
                            if isinstance(prog, dict):
                                sims = prog.get('sims', 0)
                                w_best = prog.get('best_cost', float('inf'))
                                if w_best < live_best:
                                    live_best = w_best
                            else:
                                sims = prog
                                
                            total_sims += sims
                            
                            if sims > 0:
                                status_parts.append(f"W{wid+1}:{sims//1000}k")
                            else:
                                status_parts.append(f"W{wid+1}:--")
                                
                        status_str = " ".join(status_parts)
                        
                        best_str = f"{live_best/1e6:.4f}M$" if live_best != float('inf') else "---"
                        print(f"   > [Live {m:02d}:{s:02d}] Best: {best_str} | Sims: {total_sims/1000:.1f}k | {status_str}")
                    
                    time.sleep(1.0)

                for wid, res in async_results:
                    try:
                        c, sol, _, sims_done = res.get()
                        if sol is not None:
                            epoch_results.append((c, sol))
                    except Exception as e:
                        print(f"     [Error] Worker {wid+1} crashed: {e}")

            # --- ГІЛКА ПОСЛІДОВНОГО ВИКОНАННЯ (Sequential Fallback) ---
            else:
                for t in tasks:
                    try:
                        c, sol, _, sims_done = self.worker_task(t)
                        if sol is not None:
                            epoch_results.append((c, sol))
                        print(f"   > Worker {t[6]+1} Finished. Best: {c/1e6:.4f}M$")
                    except Exception as e:
                        print(f"     [Error] Sequential Worker {t[6]+1} crashed: {e}")

            # --- ОНОВЛЕННЯ ГЛОБАЛЬНОГО АРХІВУ (Diverse Archive Logic) ---
            if epoch_results:
                epoch_results_sorted = sorted(epoch_results, key=lambda x: x[0])
                best_epoch_c, best_epoch_sol = epoch_results_sorted[0]
                
                if best_epoch_c < global_best_cost:
                    global_best_cost = best_epoch_c
                    global_best_sol = best_epoch_sol
                    print(f"\n 🏆 [EPOCH {epoch+1}] NEW GLOBAL BEST: {global_best_cost/1e6:.4f}M$ 🏆\n")

                # ВАША ІДЕАЛЬНА ЛОГІКА ДИВЕРСИФІКАЦІЇ
                elite_archive = [epoch_results_sorted[0]]
                diverse_archive = []
                req_dist = self.ctx.num_pipes // 10

                for cost, sol in epoch_results_sorted[1:]:
                    min_dist = min(
                        (sum(a != b for a, b in zip(sol, s)) for _, s in elite_archive + diverse_archive),
                        default=float('inf')
                    )
                    if min_dist > req_dist and len(diverse_archive) < 4:
                        diverse_archive.append((cost, sol))
                    elif len(elite_archive) < 2 and min_dist <= req_dist:
                        elite_archive.append((cost, sol))

                global_archive = elite_archive + diverse_archive

        print("\n[FINAL POLISH] Polishing global best solution...")
        if global_best_sol:
            polished = self.ls.gradient_squeeze(global_best_sol, max_passes=None, quick_mode=False, dyn_bonus=global_best_cost * 0.001)
            p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
            
            if p_p >= self.ctx.simulator.config.h_min and p_cost < global_best_cost:
                global_best_cost = p_cost
                global_best_sol = polished
                print(f"   > [POLISH] Improved! Final: {global_best_cost/1e6:.4f}M$")
        else:
            print("\n[WARNING] No valid solution was found before stopping. Returning safe default.")
            global_best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            try: global_best_cost, _, _, _ = self.ctx.get_cached_stats(global_best_sol)
            except: global_best_cost = float('inf')
        
        total_time = time.time() - start_time
        if global_best_cost != float('inf'):
            print(f"\n[AnalyticalSolver] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Total Time: {total_time/60:.1f}m | Total Sims: {self.ctx.sim_count:,})")
        else:
            print(f"\n[AnalyticalSolver] EXECUTION ABORTED. No valid solutions.")
        
        real_diams = [self.ctx.diameters[i] for i in global_best_sol] if global_best_sol else []
        return real_diams