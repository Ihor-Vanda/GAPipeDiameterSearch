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
            real_diams = [self.diameters[i] for i in indices]
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
            res = self.simulator.get_stats(indices) # water_sim очікує індекси!
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
        self.tabu_fingerprints = set()
        self.kick_tabu_set = set()

    def add_to_tabu(self, indices, cost):
        self.tabu_fingerprints.add(self.ctx.get_fingerprint(indices, cost))

    def is_tabu(self, indices, cost):
        return self.ctx.get_fingerprint(indices, cost) in self.tabu_fingerprints

    def hamming_distance(self, sol1, sol2):
        return sum(1 for a, b in zip(sol1, sol2) if a != b)

    def clear_all(self):
        self.active_pool.clear()
        self.tabu_fingerprints.clear()
        self.kick_tabu_set.clear()

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
        cost, _, _, crit_node = self.ctx.get_cached_stats(indices)
        best_sol = list(indices)
        if not crit_node or crit_node == "ERR": return best_sol

        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return best_sol
        
        unit_losses = self.ctx.get_cached_heuristics(indices)
        lazy_pipes = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i])
        
        candidates = []
        up_limit = max(5, int(len(path_pipes) * 0.5))
        down_limit = max(10, self.ctx.num_pipes // 3)
        
        for up_pipe in path_pipes[-up_limit:]:
            u_idx = best_sol[up_pipe] 
            if u_idx == self.ctx.max_d_idx: continue
            
            for down_pipe in lazy_pipes[:down_limit]:
                if up_pipe == down_pipe: continue
                d_idx = best_sol[down_pipe] 
                if d_idx == 0: continue
                
                test_sol = list(best_sol)
                test_sol[up_pipe] += 1
                test_sol[down_pipe] -= 1
                
                c, p, feas, _ = self.ctx.get_cached_stats(test_sol)
                
                if feas and p >= self.ctx.simulator.config.h_min:
                    surplus = p - self.ctx.simulator.config.h_min
                    score = c - (surplus * dyn_bonus) 
                    candidates.append((score, test_sol, {up_pipe}))

        for up_pipe in path_pipes[-int(up_limit*0.7):]:
            u_idx = best_sol[up_pipe]
            if u_idx == self.ctx.max_d_idx: continue
            for d1, d2 in itertools.combinations(lazy_pipes[:int(down_limit*0.7)], 2):
                if up_pipe in (d1, d2): continue
                d1_idx, d2_idx = best_sol[d1], best_sol[d2]
                if d1_idx == 0 or d2_idx == 0: continue
                
                test_sol = list(best_sol)
                test_sol[up_pipe] += 1
                test_sol[d1] -= 1
                test_sol[d2] -= 1
                
                c, p, feas, _ = self.ctx.get_cached_stats(test_sol)
                if feas and p >= self.ctx.simulator.config.h_min:
                    surplus = p - self.ctx.simulator.config.h_min
                    score = c - (surplus * dyn_bonus)
                    candidates.append((score, test_sol, {up_pipe}))

        if not candidates: return best_sol
        candidates.sort(key=lambda x: x[0])
        best_score, best_candidate_sol, locked = candidates[0]
        
        squeezed = self.gradient_squeeze(best_candidate_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
        sq_c, _, _, _ = self.ctx.get_cached_stats(squeezed)
        
        if sq_c < cost: return squeezed
        return best_sol

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
        # Базові ваги обернено пропорційні діаметрам
        for u, v in temp_G.edges():
            idx = self.ctx.edge_to_pipe[(u, v)]
            temp_G[u][v]['weight'] = 100.0 / (indices[idx] + 1)
            
        candidates = []
        # Шукаємо 15 топологічно різних шляхів через штрафування
        for _ in range(15):
            try:
                path = nx.shortest_path(temp_G, source, crit_node, weight='weight')
                if path != dom_nodes:
                    candidates.append(path)
                # Штрафуємо ребра знайденого шляху в 5 разів, щоб наступний пошук обійшов їх
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
        
        combo_limit = min(30, max(10, self.ctx.num_pipes // 3))
        all_pairs = list(itertools.combinations(periphery_pipes[:combo_limit], 2))
        random.shuffle(all_pairs)
        
        for d1, d2 in all_pairs[:60]: 
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
    
    def diameter_diversity_kick(self, indices, stagnation_counter): # ПАТЧ: додали stagnation_counter
        from collections import Counter
        import random
        
        diam_counts = Counter(indices)
        most_common_ratio = diam_counts.most_common(1)[0][1] / self.ctx.num_pipes
        
        if most_common_ratio < 0.40:
            return None, None, "" 
            
        kicked, locked = list(indices), set()
        
        # ПАТЧ: Адаптивна інтенсивність! Чим довше застрягли, тим більше труб міняємо
        base_n = max(3, self.ctx.num_pipes // 8)
        n_to_change = min(self.ctx.num_pipes // 2, base_n + int(stagnation_counter * 0.5))
        
        pipes_to_change = random.sample(range(self.ctx.num_pipes), n_to_change)
        
        for i in pipes_to_change:
            target_bucket = (kicked[i] + self.ctx.max_d_idx // 2) % (self.ctx.max_d_idx + 1)
            kicked[i] = target_bucket
            locked.add(i)
            
        if not locked: return None, None, ""
            
        healed, is_feas, _ = self.ls.heal_network(kicked, locked)
        if not is_feas: return None, None, ""
            
        return healed, locked, f"DIAM-DIVERSITY: Injected {len(locked)} varied diams (Intensity: {n_to_change})"
    
    def ruin_and_recreate_kick(self, indices, stagnation_counter):
        """
        LNS (Large Neighborhood Search): Вирізає кластер навколо критичного вузла
        і жадібно відбудовує його з нуля.
        """
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        # Адаптивний радіус руйнування (від 1 до 3 ребер навколо вузла)
        radius = 1 if stagnation_counter < 5 else (2 if stagnation_counter < 12 else 3)
        
        try:
            # BFS пошук сусідів
            cluster_nodes = list(nx.single_source_shortest_path_length(self.ctx.base_G_flow, crit_node, cutoff=radius).keys())
        except:
            return None, None, ""

        cluster_pipes = set()
        for u in cluster_nodes:
            for p in self.ctx.node_to_pipes.get(u, []):
                cluster_pipes.add(p)

        if not cluster_pipes: return None, None, ""

        kicked = list(indices)
        ruined_count = 0
        
        # RUIN (Руйнування): Скидаємо всі труби кластера в мінімальний діаметр (0)
        for p in cluster_pipes:
            if kicked[p] > 0:
                kicked[p] = 0
                ruined_count += 1

        if ruined_count == 0: return None, None, ""

        # RECREATE (Відбудова): Дозволяємо heal_network жадібно підняти ТІЛЬКИ ті труби, які дійсно потрібні
        # Важливо: ми передаємо порожній set(), щоб heal міг змінювати зруйновані труби
        healed_sol, is_feas, boosts = self.ls.heal_network(kicked, set())
        
        if not is_feas: return None, None, ""

        # Повертаємо cluster_pipes як 'locked', щоб gradient_squeeze тимчасово не чіпав нашу нову відбудову
        return healed_sol, cluster_pipes, f"R&R (LNS): Ruined {ruined_count} pipes (Rad: {radius}). Rebuilt {boosts}x."

# =============================================================================
# 5. ORCHESTRATOR (Parallel Island Model Architecture)
# =============================================================================
class AnalyticalSolver:
    worker_task = None 
    
    def __init__(self, simulator, available_diameters, v_opt=1.0, max_iters=10, 
                 time_limit_sec=None, sim_budget=None, quality_target=None, pool=None, log_dir=".", n_workers=None):
        self.ctx = SolverContext(simulator, available_diameters, v_opt)
        self.pool = SolutionPool(self.ctx)
        self.ls = LocalSearch(self.ctx)
        self.kicker = KickStrategies(self.ctx, self.ls)
        self.mp_pool = pool 
        self.log_dir = log_dir
        
        self.n_workers = n_workers or max(1, multiprocessing.cpu_count() - 1)
        self.max_iters = max_iters
        self.best_found_solution = None
        
        self.network_class = self._classify_network()
        self._configure_for_scale()
        
        if sim_budget is not None:
            self.max_sims = sim_budget
        else:
            self.max_sims = self.BASE_SIM_BUDGET
            
        if time_limit_sec is not None:
            self.time_limit_sec = time_limit_sec
        else:
            self.time_limit_sec = 3600

        print(f"[Config] Network: {self.network_class} ({self.ctx.num_pipes} pipes)")
        print(f"  Target Budget: {self.max_sims:,} sims (≈{self.time_limit_sec/60:.1f} min)")
        print(f"  BeamWidth={self.BEAM_WIDTH}, Restarts={self.N_RESTARTS}")

        self.strategies = ["TOPO-DIV", "LOOP_BALANCE", "SYNC_TRIM", "FINISHER", "BOTTLENECK", "SHOCK"]
        if nx.is_tree(self.ctx.base_G_flow):
            self.strategies.remove("LOOP_BALANCE")
        self.strategies.append("DIAM_DIVERSITY")
        self.strategies.append("RUIN_RECREATE") # ПАТЧ: Додали нову стратегію
        
        self.strat_wins = {s: 1.0 for s in self.strategies}
        self.strat_tries = {s: 1.0 for s in self.strategies}

    def _classify_network(self):
        n = self.ctx.num_pipes
        if n <= 50: return "SMALL"
        if n <= 200: return "MEDIUM"
        if n <= 500: return "LARGE"
        return "XLARGE"

    def _configure_for_scale(self):
        n = self.ctx.num_pipes
        cls = self.network_class
        
        self.BEAM_WIDTH = {"SMALL": max(5, n//10), "MEDIUM": 5, "LARGE": 5, "XLARGE": 5}[cls]
        self.SINGLE_CANDIDATES = {"SMALL": max(6, n//5), "MEDIUM": max(6, n//10), "LARGE": max(6, n//20), "XLARGE": 10}[cls]
        self.N_RESTARTS = {"SMALL": 4, "MEDIUM": 3, "LARGE": 2, "XLARGE": 1}[cls]
        
        # ЗБІЛЬШЕНО БЮДЖЕТИ:
        self.BASE_SIM_BUDGET = {
            "SMALL": 30000, 
            "MEDIUM": 100000, 
            "LARGE": 1000000, # 1 Мільйон для Balerma!
            "XLARGE": 3000000
        }[cls]

    def _calculate_ideal_d(self, flow, target_v):
        if abs(flow) < 1e-6: return 0 # Повертає індекс
        d_ideal = math.sqrt((4.0 * abs(flow)) / (math.pi * target_v))
        pos = bisect.bisect_left(self.ctx.diameters, d_ideal)
        if pos < len(self.ctx.diameters): return pos
        return self.ctx.max_d_idx

    def _make_diverse_seeds(self):
        seeds = []
        for v in [1.0, 1.2, 0.8]:
            idx_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            for _ in range(self.max_iters):
                # ВИПРАВЛЕННЯ: get_hydraulic_state вимагає фізичних метрів, а не індексів!
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
        """
        PATH-RELINKING: Плавно перетворює одне хороше рішення на інше,
        збираючи валідні проміжні стани як нові насіння.
        """
        import random
        import itertools
        seeds = []
        
        # Якщо в архіві менше 2 рішень, перелінкування неможливе
        if len(archive) < 2:
            return []

        # Беремо Топ-4 рішення і створюємо всі можливі пари (А, Б)
        top_sols = [x[1] for x in archive[:4]]
        pairs = list(itertools.combinations(top_sols, 2))
        random.shuffle(pairs)

        for sol_A, sol_B in pairs:
            # Знаходимо індекси труб, які відрізняються
            diff_indices = [i for i in range(self.ctx.num_pipes) if sol_A[i] != sol_B[i]]
            random.shuffle(diff_indices) # Випадковий порядок заміни

            current_path_sol = list(sol_A)
            
            # Крокуємо від А до Б
            for idx in diff_indices:
                current_path_sol[idx] = sol_B[idx]
                
                # Перевіряємо проміжний стан
                _, p, feas, _ = self.ctx.get_cached_stats(current_path_sol)
                if feas and p >= self.ctx.simulator.config.h_min:
                    # Якщо проміжний крок валідний - зжимаємо його і беремо в роботу!
                    squeezed = self.ls.gradient_squeeze(current_path_sol, quick_mode=True)
                    seeds.append(squeezed)
                    
                if len(seeds) >= self.BEAM_WIDTH:
                    break
            if len(seeds) >= self.BEAM_WIDTH:
                break
                
        # Якщо перелінкування не дало достатньо валідних кроків, добиваємо лідерами
        while len(seeds) < self.BEAM_WIDTH:
            seeds.append(list(top_sols[0]))
            
        return seeds

    def _make_warm_seeds(self, archive, worker_id=0):
        import random
        seeds = []
        
        # ДЕТЕРМІНОВАНИЙ РОЗПОДІЛ ДЛЯ МАЛИХ ПУЛІВ
        if worker_id == 0: role = "EXPLOITER"
        elif worker_id == 1: role = "RELINKER"   # ПАТЧ: Worker 1 тепер робить Path-Relinking!
        elif worker_id == 2: role = "EXPLORER"
        elif worker_id == 3: role = "ADVENTURER"
        else:
            # ЙМОВІРНІСНИЙ РОЗПОДІЛ ДЛЯ ВЕЛИКИХ ПУЛІВ
            caste_roll = random.random()
            if caste_roll < 0.30: role = "EXPLOITER"
            elif caste_roll < 0.60: role = "RELINKER" # 30% воркерів зшивають топології
            elif caste_roll < 0.85: role = "EXPLORER"
            else: role = "ADVENTURER"
            
        # Застосування ролей
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
            
        # Генерація збурених рішень (для EXPLOITER та EXPLORER)
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
                seeds.append(base_sol) # Захист від infeasible fallback
            
        return seeds

    def _run_single_search(self, seeds, time_budget, global_best_cost, worker_id=None, shared_progress=None):
        start_time = time.time()
        self.pool.clear_all()
        
        best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds])
        dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001 
        
        valid_sols = []
        for s in seeds:
            c, p, feas, _ = self.ctx.get_cached_stats(s)
            p_surplus = p - self.ctx.simulator.config.h_min
            score = c - (p_surplus * dyn_bonus)
            self.pool.active_pool.append((score, c, s))
            self.pool.add_to_tabu(s, c)
            if p >= self.ctx.simulator.config.h_min:
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
        while True:
            if shared_progress is not None and worker_id is not None:
                shared_progress[worker_id] = {
                    "round": round_idx + 1,
                    "sims": self.ctx.sim_count,
                    "best_cost": run_best_cost
                }
            
            elapsed = time.time() - start_time
            if elapsed > time_budget or self.ctx.sim_count >= self.max_sims: 
                break
                
            dyn_bonus = min(run_best_cost, global_best_cost) * 0.001
            just_flushed = False
            
            progress_ratio = min(1.0, self.ctx.sim_count / max(1, self.max_sims))
            is_late_game = progress_ratio > 0.5 
            stag_limit = 3 + int(9 * progress_ratio) 
            
            if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()
            next_gen = []
            
            if round_idx % 3 == 0 or stagnation_counter == 0:
                swapped = self.ls.swap_search(run_best_sol, dyn_bonus)
                c, p, _, _ = self.ctx.get_cached_stats(swapped)
                if c < run_best_cost:
                    is_ghost = False
                    if (run_best_cost - c) > (run_best_cost * 0.02):
                        is_ghost = self.ctx.is_ghost_solution(swapped, c)
                        
                    if is_ghost:
                        self.ctx.log(f"   > [SHIELD] Swap Heuristic illusion blocked ({c/1e6:.4f}M$)!")
                    else:
                         diff = run_best_cost - c
                         run_best_cost, run_best_sol = c, swapped
                         self.ctx.log(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$)")
                         stagnation_counter = 0
                         self.pool.kick_tabu_set.clear()
                         p_surplus = p - self.ctx.simulator.config.h_min
                         self.pool.active_pool.insert(0, (c - (p_surplus * dyn_bonus) - (run_best_cost * 0.1), c, swapped))

            if stagnation_counter >= 1 or round_idx % 2 == 0:
                n_total = sum(self.strat_tries.values())
                strategy = max(self.strategies, key=lambda s: (self.strat_wins[s] / self.strat_tries[s]) + math.sqrt(2 * math.log(n_total) / self.strat_tries[s]))
                self.strat_tries[strategy] += 1
                
                self.ctx.log(f"[FORCE] Applying '{strategy}'...")
                
                kick_mode = round_idx % 3
                if kick_mode == 0: kick_target = run_best_sol
                elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
                else:
                    pool_sols = [x[2] for x in self.pool.active_pool[:3]]
                    kick_target = random.choice(pool_sols) if pool_sols else run_best_sol
                
                forced_sol, locked, path_sig = None, None, None
                
                # ПАТЧ: Фіксуємо витрати симуляцій до виклику стратегії
                sims_before_kick = self.ctx.sim_count
                
                if strategy == "SHOCK": forced_sol, locked, log_msg = self.kicker.forcing_hand_kick(kick_target)
                elif strategy == "BOTTLENECK": forced_sol, locked, log_msg = self.kicker.upstream_bottleneck_kick(kick_target)
                elif strategy == "LOOP_BALANCE": forced_sol, locked, log_msg = self.kicker.loop_balancing_kick(kick_target, dyn_bonus)
                elif strategy == "DIAM_DIVERSITY": forced_sol, locked, log_msg = self.kicker.diameter_diversity_kick(kick_target, stagnation_counter) # ПАТЧ
                elif strategy == "RUIN_RECREATE": forced_sol, locked, log_msg = self.kicker.ruin_and_recreate_kick(kick_target, stagnation_counter) # ПАТЧ
                elif strategy == "FINISHER": forced_sol, locked, log_msg = self.kicker.micro_trim_kick(kick_target)
                elif strategy == "SYNC_TRIM": forced_sol, locked, log_msg = self.kicker.sync_trim_kick(kick_target, dyn_bonus)
                else: forced_sol, locked, log_msg, path_sig = self.kicker.topological_inversion_kick(kick_target, self.pool.kick_tabu_set)
                
                if forced_sol and locked:
                    self.ctx.log(f"     -> {log_msg}")
                    starved_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=8, dyn_bonus=dyn_bonus)
                    final_sol = self.ls.gradient_squeeze(starved_sol, dyn_bonus=dyn_bonus)
                    c, p, _, _ = self.ctx.get_cached_stats(final_sol)
                    
                    if p >= self.ctx.simulator.config.h_min:
                        p_surplus = p - self.ctx.simulator.config.h_min
                        score = c - (p_surplus * dyn_bonus)
                        
                        if stagnation_counter >= stag_limit:
                            self.ctx.log("     [INFO] Partial flush to force exploration (keeping best)!")
                            if self.pool.active_pool:
                                best_in_pool = min(self.pool.active_pool, key=lambda x: x[1])
                                self.pool.active_pool = [best_in_pool]
                            just_flushed = True 
                            
                        self.pool.active_pool.insert(0, (score, c, final_sol))
                        
                        # ПАТЧ (ALNS): Розрахунок ROI (Return on Investment)
                        sims_used = max(1, self.ctx.sim_count - sims_before_kick)
                        # Множник: еталон 20 симуляцій = множник 1.0. Витратив 100 симуляцій = множник 0.2
                        efficiency_multiplier = min(2.0, 20.0 / sims_used) 
                        
                        if c < run_best_cost:
                            is_ghost = False
                            if (run_best_cost - c) > (run_best_cost * 0.02):
                                is_ghost = self.ctx.is_ghost_solution(final_sol, c)
                                
                            if is_ghost:
                                self.ctx.log(f"     [SHIELD] Force Heuristic illusion blocked ({c/1e6:.4f}M$)!")
                            else:
                                diff = run_best_cost - c
                                run_best_cost, run_best_sol = c, final_sol
                                # Нагорода нормалізована!
                                self.strat_wins[strategy] += (3.0 * efficiency_multiplier) 
                                self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$) | ROI: {efficiency_multiplier:.2f}x")
                        elif c <= run_best_cost * 1.001: 
                            self.strat_wins[strategy] += (0.5 * efficiency_multiplier) 
                            self.ctx.log(f"     -> Injection Acceptable: {c/1e6:.4f}M$")
                        else:
                            self.ctx.log(f"     -> Injection Poor: {c/1e6:.4f}M$ (No Reward)")
                            
                        if path_sig: self.pool.kick_tabu_set.add(path_sig)

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
                    s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "upgrade", dyn_bonus)
                    if sol: next_gen.append((s, c, sol))
                
                for pipe_idx in low_friction[:8]:
                     s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "downgrade", dyn_bonus)
                     if sol: next_gen.append((s, c, sol))
                     
                if len(high_friction) >= 2:
                    for p1, p2 in itertools.combinations(high_friction[:5], 2): 
                        s, c, sol = self.ls.evaluate_candidate(parent_sol, [p1, p2], "upgrade", dyn_bonus)
                        if sol: next_gen.append((s, c, sol))
            
            if not next_gen:
                if not just_flushed: stagnation_counter += 1
                round_idx += 1
                continue

            next_gen.sort(key=lambda x: x[0]) 
            unique_next_pool = []
            found_new_record = False
            
            max_d = max(2, self.ctx.num_pipes // 7) 
            min_dist = max(1, int(max_d * (1.0 - (progress_ratio ** 2))))
            
            for rank, (score, cost, sol) in enumerate(next_gen):
                if self.pool.is_tabu(sol, cost): continue
                if len(unique_next_pool) < self.BEAM_WIDTH:
                    if rank == 0: refined_sol = self.ls.gradient_squeeze(sol, dyn_bonus=dyn_bonus)
                    elif rank < 3: refined_sol = self.ls.gradient_squeeze(sol, max_passes=4, quick_mode=not is_late_game, dyn_bonus=dyn_bonus)
                    else: refined_sol = sol 

                    is_diverse = True
                    for _, _, peer_sol in unique_next_pool:
                        if self.pool.hamming_distance(refined_sol, peer_sol) < min_dist:
                            is_diverse = False; break
                            
                    if not is_diverse and not just_flushed: continue

                    real_cost, p_min, _, _ = self.ctx.get_cached_stats(refined_sol)
                    if p_min >= self.ctx.simulator.config.h_min:
                        real_p_surplus = p_min - self.ctx.simulator.config.h_min
                        unique_next_pool.append((real_cost - (real_p_surplus * dyn_bonus), real_cost, refined_sol))
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

            if found_new_record or just_flushed:
                stagnation_counter = 0; self.pool.kick_tabu_set.clear()
            else:
                stagnation_counter += 1
                if stagnation_counter % 8 == 0: self.pool.kick_tabu_set.clear()

            if unique_next_pool:
                unique_next_pool.sort(key=lambda x: x[0])
                self.pool.active_pool = unique_next_pool[:self.BEAM_WIDTH]
                
            round_idx += 1
                
        if shared_progress is not None and worker_id is not None:
            shared_progress[worker_id] = {"round": "DONE", "sims": self.ctx.sim_count, "best_cost": run_best_cost}

        return run_best_cost, run_best_sol

    def solve_standalone(self):
        start_time = time.time()
        
        global_archive = [] 
        global_best_cost = float('inf')
        global_best_sol = None

        if self.mp_pool and AnalyticalSolver.worker_task:
            print(f"[AnalyticalSolver] ⚡ Initiating PARALLEL Island Model Search...")
            
            manager = multiprocessing.Manager() 
            epochs = 2
            workers = getattr(self, 'n_workers', max(1, multiprocessing.cpu_count() - 1))
            time_per_epoch = self.time_limit_sec / epochs
            
            try:
                for epoch in range(epochs):
                    if time.time() - start_time >= self.time_limit_sec: break
                    
                    print(f"\n==============================================")
                    print(f" [EPOCH {epoch+1}/{epochs}] Parallel Workers: {workers} | Time: {time_per_epoch/60:.1f} min")
                    print(f"==============================================")
                    
                    shared_progress = manager.dict()
                    async_results = []
                    
                    for w_idx in range(workers):
                        seed_modifier = epoch * 10 + w_idx
                        task_args = (
                            self.ctx.diameters, 
                            self.ctx.v_opt, 
                            time_per_epoch, 
                            global_best_cost, 
                            global_archive, 
                            seed_modifier, 
                            w_idx, 
                            shared_progress, 
                            self.log_dir, 
                            epoch
                        )
                        res = self.mp_pool.apply_async(AnalyticalSolver.worker_task, (task_args,))
                        async_results.append((w_idx, res))
                    
                    epoch_start_time = time.time()
                    last_print_time = 0
                    
                    while True:
                        all_done = all(res.ready() for _, res in async_results)
                        if all_done: break
                            
                        curr_time = time.time()
                        if curr_time - last_print_time >= 30.0:
                            last_print_time = curr_time
                            elapsed_total = curr_time - epoch_start_time
                            mins, secs = divmod(int(elapsed_total), 60)
                            
                            w_stats = []
                            live_sims = 0
                            current_live_best = global_best_cost 
                            
                            for w_idx in range(workers):
                                prog = shared_progress.get(w_idx, {})
                                if prog:
                                    rnd = prog.get('round', 0)
                                    live_sims += prog.get('sims', 0)
                                    
                                    w_best = prog.get('best_cost', float('inf'))
                                    if w_best < current_live_best:
                                        current_live_best = w_best
                                        
                                    status = "✔" if str(rnd) == "DONE" else str(rnd)
                                    w_stats.append(f"W{w_idx+1}:{status}")
                                else:
                                    w_stats.append(f"W{w_idx+1}:--")
                                    
                            w_str = " ".join(w_stats)
                            best_val = f"{current_live_best/1e6:.4f}M$" if current_live_best != float('inf') else "---"
                            
                            print(f"   > [Live {mins:02d}:{secs:02d}] Best: {best_val} | Sims: {live_sims/1000:.1f}k | {w_str}")
                    
                    epoch_sims = 0
                    req_dist = max(2, self.ctx.num_pipes // 10) 
                    
                    for w_idx, res in async_results:
                        res_cost, res_sol, _, sims_used = res.get()
                        epoch_sims += sims_used
                        self.ctx.sim_count += sims_used 
                        
                        if res_sol:
                            check_cost, check_p, check_feas, _ = self.ctx.get_cached_stats(res_sol)
                            
                            if check_feas and check_p >= self.ctx.simulator.config.h_min:
                                is_diverse = True
                                for arc_cost, arc_sol in global_archive:
                                    dist = sum(1 for a, b in zip(res_sol, arc_sol) if a != b)
                                    if dist < req_dist:
                                        is_diverse = False
                                        break
                                
                                if is_diverse or check_cost < global_best_cost:
                                    global_archive.append((check_cost, res_sol))
                            else:
                                print(f"   > [WARNING] Worker {w_idx+1:02d} returned a Ghost Solution (P={check_p:.2f}m). Rejected!")
                    
                    global_archive.sort(key=lambda x: x[0])
                    pruned_archive = []
                    
                    for cost, sol in global_archive:
                        if not pruned_archive:
                            pruned_archive.append((cost, sol))
                        else:
                            conflict = False
                            for p_cost, p_sol in pruned_archive:
                                dist = sum(1 for a, b in zip(sol, p_sol) if a != b)
                                if dist < req_dist:
                                    conflict = True
                                    break
                            
                            if not conflict or cost == pruned_archive[0][0]: 
                                pruned_archive.append((cost, sol))
                                
                        if len(pruned_archive) >= 5: 
                            break
                        
                    global_archive = pruned_archive
                    
                    if global_archive and global_archive[0][0] < global_best_cost:
                        global_best_cost = global_archive[0][0]
                        global_best_sol = global_archive[0][1]
                        print(f"\n 🏆 [EPOCH {epoch+1}] NEW GLOBAL BEST: {global_best_cost/1e6:.4f}M$ 🏆\n")

            except KeyboardInterrupt:
                print("\n[AnalyticalSolver] 🛑 User requested early stop (Ctrl+C). Halting parallel execution...")
                
        else:
            print(f"[AnalyticalSolver] 🐢 Initiating SEQUENTIAL Search (No Pool)...")
            time_per_restart = self.time_limit_sec / self.N_RESTARTS
            
            try:
                for restart_idx in range(self.N_RESTARTS):
                    if time.time() - start_time >= self.time_limit_sec: break
                    print(f"\n==============================================")
                    print(f" [RESTART {restart_idx+1}/{self.N_RESTARTS}] Budget: {time_per_restart/60:.1f} min")
                    print(f"==============================================")
                    
                    random.seed(restart_idx * 137 + 42)
                    np.random.seed(restart_idx * 137 + 42)
                    
                    seeds = self._make_diverse_seeds() if not global_archive else self._make_warm_seeds(global_archive, worker_id=restart_idx)
                    res_cost, res_sol = self._run_single_search(seeds, time_per_restart, global_best_cost)
                    
                    global_archive.append((res_cost, res_sol))
                    global_archive.sort(key=lambda x: x[0])
                    global_archive = global_archive[:5] 
                    
                    if res_cost < global_best_cost:
                        global_best_cost = res_cost
                        global_best_sol = res_sol
                        print(f"\n 🏆 [RESTART {restart_idx+1}] NEW GLOBAL BEST: {global_best_cost/1e6:.4f}M$ 🏆\n")
            except KeyboardInterrupt:
                print("\n[AnalyticalSolver] 🛑 User requested early stop (Ctrl+C). Halting sequential execution...")

        if global_best_sol is not None:
            print(f"\n[FINAL POLISH] Polishing global best solution...")
            polished = self.ls.gradient_squeeze(global_best_sol, max_passes=None, quick_mode=False, dyn_bonus=global_best_cost * 0.001)
            p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
            
            if p_p >= self.ctx.simulator.config.h_min and p_cost < global_best_cost:
                global_best_cost = p_cost
                global_best_sol = polished
                print(f"   > [POLISH] Improved! Final: {global_best_cost/1e6:.4f}M$")
        else:
            print("\n[WARNING] No valid solution was found before stopping. Returning safe default.")
            global_best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            try:
                global_best_cost, _, _, _ = self.ctx.get_cached_stats(global_best_sol)
            except:
                global_best_cost = float('inf')
        
        total_time = time.time() - start_time
        if global_best_cost != float('inf'):
            print(f"\n[AnalyticalSolver] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Total Time: {total_time/60:.1f}m | Total Sims: {self.ctx.sim_count:,})")
        else:
            print(f"\n[AnalyticalSolver] EXECUTION ABORTED (Total Time: {total_time/60:.1f}m)")
            
        # КОНВЕРТАЦІЯ ФІНАЛЬНОГО РЕЗУЛЬТАТУ ДЛЯ КОРИСТУВАЧА
        real_diameters_sol = [self.ctx.diameters[i] for i in global_best_sol] if global_best_sol else None
        
        self.best_found_solution = real_diameters_sol
        return real_diameters_sol