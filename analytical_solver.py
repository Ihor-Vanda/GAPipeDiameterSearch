import math
import random
import sys
import time
import bisect
import numpy as np
import itertools
import networkx as nx

# =============================================================================
# 1. SHARED CONTEXT 
# =============================================================================
class SolverContext:
    def __init__(self, simulator, available_diameters, v_opt=1.0):
        self.simulator = simulator
        self.diameters = sorted(available_diameters)
        self.v_opt = v_opt
        self.num_pipes = simulator.n_variables
        
        self.diam_to_idx = {d: i for i, d in enumerate(self.diameters)}
        # ВИПРАВЛЕННЯ 7: O(1) словник для імен компонентів
        self.comp_name_to_idx = {name: i for i, name in enumerate(simulator.component_names)}
        
        self.sim_cache = {}
        self.heuristic_cache = {}
        self.MAX_CACHE_SIZE = 50000 
        
        self.base_G_flow = nx.Graph()
        self.edge_to_pipe = {}
        for u, v, k in self.simulator.graph.edges(keys=True):
            if k in self.simulator.component_names:
                idx = self.comp_name_to_idx[k]
                self.base_G_flow.add_edge(u, v)
                self.edge_to_pipe[(u, v)] = idx
                self.edge_to_pipe[(v, u)] = idx

        self.lengths = self.simulator.lengths
        self.costs_array = self.simulator.costs 

    def log(self, message):
        print(f"   {message}")

    def get_fingerprint(self, solution, known_cost=None):
        sol_tuple = tuple(solution)
        if known_cost is not None:
            cost_bucket = int(known_cost / 50) * 50 
            return (cost_bucket, sol_tuple)
        idx = self.get_current_indices(solution)
        cost, _, _, _ = self.get_cached_stats(idx)
        return (int(cost / 50) * 50, sol_tuple)

    def get_current_indices(self, solution):
        return [self.diam_to_idx.get(d, len(self.diameters)-1) for d in solution]
    
    def get_cached_stats(self, indices):
        if len(self.sim_cache) > self.MAX_CACHE_SIZE: self.sim_cache.clear()
        sig = tuple(indices)
        if sig not in self.sim_cache:
            self.sim_cache[sig] = self.simulator.get_stats(indices)
        return self.sim_cache[sig]
        
    def get_cached_heuristics(self, indices):
        if len(self.heuristic_cache) > self.MAX_CACHE_SIZE: self.heuristic_cache.clear()
        sig = tuple(indices)
        if sig not in self.heuristic_cache:
            self.heuristic_cache[sig] = self.simulator.get_heuristics(indices)
        return self.heuristic_cache[sig]

    def get_dominant_path(self, curr_indices, crit_node):
        try:
            for u, v in self.base_G_flow.edges():
                idx = self.edge_to_pipe[(u, v)]
                self.base_G_flow[u][v]['weight'] = 100.0 / (curr_indices[idx] + 1)
                    
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

    def get_lazy_periphery(self, solution):
        curr_indices = self.get_current_indices(solution)
        _, _, _, crit_node = self.get_cached_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return curr_indices, []
        path_pipes, _ = self.get_dominant_path(curr_indices, crit_node)
        unit_losses = self.get_cached_heuristics(curr_indices)
        threshold_idx = int(len(self.diameters) * 0.6)
        periphery_pipes = [i for i in range(self.num_pipes) if i not in path_pipes and curr_indices[i] <= threshold_idx]
        periphery_pipes.sort(key=lambda i: unit_losses[i])
        return curr_indices, periphery_pipes

# =============================================================================
# 2. SOLUTION POOL MANAGER (З контролем різноманітності Хемінга)
# =============================================================================
class SolutionPool:
    def __init__(self, ctx):
        self.ctx = ctx
        self.active_pool = []
        self.tabu_fingerprints = set()
        self.kick_tabu_set = set()

    def add_to_tabu(self, solution, cost):
        self.tabu_fingerprints.add(self.ctx.get_fingerprint(solution, cost))

    def is_tabu(self, solution, cost):
        return self.ctx.get_fingerprint(solution, cost) in self.tabu_fingerprints

    # ВИПРАВЛЕННЯ 2 та 5: Дистанція Хемінга для захисту різноманіття пулу
    def hamming_distance(self, sol1, sol2):
        return sum(1 for a, b in zip(sol1, sol2) if a != b)

    def is_diverse_enough(self, candidate_sol, min_distance=2):
        """Перевіряє, чи не є рішення клоном вже існуючих у пулі."""
        if not self.active_pool: return True
        for _, _, pool_sol in self.active_pool:
            if self.hamming_distance(candidate_sol, pool_sol) < min_distance:
                return False # Занадто схоже на існуюче
        return True

    def clear_all(self):
        self.active_pool.clear()
        self.tabu_fingerprints.clear()
        self.kick_tabu_set.clear()

# =============================================================================
# 3. LOCAL SEARCH ENGINE 
# =============================================================================
class LocalSearch:
    def __init__(self, ctx):
        self.ctx = ctx

    def gradient_squeeze(self, solution, locked_pipes=None, max_passes=None, quick_mode=False, verbose=False, dyn_bonus=None):
        locked_pipes = locked_pipes or set()
        current_solution = list(solution)
        improved = True
        passes = 0
        active_indices = [i for i in range(self.ctx.num_pipes) if i not in locked_pipes]
        
        curr_indices = self.ctx.get_current_indices(current_solution)
        cost, p_min, _, _ = self.ctx.get_cached_stats(curr_indices)
        bonus = dyn_bonus if dyn_bonus else 50000.0 # Fallback
        
        while improved:
            improved = False
            passes += 1
            if max_passes and passes > max_passes: break
            
            curr_indices = self.ctx.get_current_indices(current_solution)
            unit_losses = self.ctx.get_cached_heuristics(curr_indices)
            savings_candidates = []
            
            for i in active_indices:
                if quick_mode and unit_losses[i] > 0.05: continue
                current_idx = curr_indices[i]
                if current_idx > 0:
                    dollar_save = self.ctx.lengths[i] * (self.ctx.costs_array[current_idx] - self.ctx.costs_array[current_idx - 1])
                    if dollar_save < 50.0: continue
                    risk = unit_losses[i] + 1e-9
                    score = dollar_save / risk
                    savings_candidates.append((i, score, dollar_save))
            
            if not savings_candidates: break
            savings_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates_to_try = savings_candidates[:5 if quick_mode else 20]
            batch_count = 0
            
            for idx, _, save in candidates_to_try:
                current_idx = curr_indices[idx] 
                test_solution = list(current_solution)
                test_solution[idx] = self.ctx.diameters[current_idx - 1]
                
                t_indices = self.ctx.get_current_indices(test_solution)
                _, t_p, t_feas, t_crit = self.ctx.get_cached_stats(t_indices)
                
                if t_feas and t_p >= self.ctx.simulator.config.h_min:
                    current_solution = test_solution
                    p_min = t_p
                    improved = True
                    batch_count += 1
                    if batch_count >= 5: break
                
                elif not quick_mode and save > (bonus * 0.5) and t_crit is not None:
                    neighbors = []
                    for u, v, k in self.ctx.simulator.graph.edges(keys=True):
                        if (u == t_crit or v == t_crit) and k in self.ctx.simulator.component_names:
                            # ВИПРАВЛЕННЯ 7: O(1) пошук імені замість .index(k)
                            n_idx = self.ctx.comp_name_to_idx[k]
                            if n_idx != idx and n_idx not in locked_pipes:
                                neighbors.append(n_idx)
                    
                    best_fix, best_net_save, best_neighbor, best_p = None, 0, -1, p_min
                    
                    for n_idx in neighbors:
                        n_curr_d = self.ctx.diam_to_idx[test_solution[n_idx]]
                        if n_curr_d < len(self.ctx.diameters) - 1:
                            fix_solution = list(test_solution)
                            fix_solution[n_idx] = self.ctx.diameters[n_curr_d + 1] 
                            expansion_cost = self.ctx.lengths[n_idx] * (self.ctx.costs_array[n_curr_d + 1] - self.ctx.costs_array[n_curr_d])
                            net_save = save - expansion_cost
                            
                            if net_save > (bonus * 0.5): 
                                f_indices = self.ctx.get_current_indices(fix_solution)
                                _, f_p, f_feas, _ = self.ctx.get_cached_stats(f_indices)
                                if f_feas and f_p >= self.ctx.simulator.config.h_min:
                                    if net_save > best_net_save:
                                        best_net_save, best_fix, best_neighbor, best_p = net_save, fix_solution, n_idx, f_p

                    if best_fix:
                        current_solution = best_fix
                        p_min = best_p
                        improved = True
                        batch_count += 1
                        if batch_count >= 5: break

        return current_solution

    def heal_network(self, solution, locked_pipes):
        kicked = list(solution)
        boosts = 0
        for _ in range(40): 
            test_indices = self.ctx.get_current_indices(kicked)
            _, t_p, t_feas, crit_node = self.ctx.get_cached_stats(test_indices)
            if t_feas and t_p >= self.ctx.simulator.config.h_min: return kicked, True, boosts
            
            path_pipes, _ = self.ctx.get_dominant_path(test_indices, crit_node)
            if not path_pipes: return kicked, False, boosts
            
            unit_losses = self.ctx.get_cached_heuristics(test_indices)
            worst_pipe, max_loss = -1, -1.0
            
            for idx in path_pipes:
                if idx in locked_pipes: continue 
                if test_indices[idx] < len(self.ctx.diameters) - 1: 
                    if unit_losses[idx] > max_loss:
                        max_loss, worst_pipe = unit_losses[idx], idx
            
            if worst_pipe != -1:
                kicked[worst_pipe] = self.ctx.diameters[test_indices[worst_pipe] + 1]
                boosts += 1
            else: break
        return kicked, False, boosts

    def swap_search(self, solution, dyn_bonus):
        curr_indices = self.ctx.get_current_indices(solution)
        cost, _, _, crit_node = self.ctx.get_cached_stats(curr_indices)
        best_sol = list(solution)
        if not crit_node or crit_node == "ERR": return best_sol

        path_pipes, _ = self.ctx.get_dominant_path(curr_indices, crit_node)
        if not path_pipes: return best_sol
        
        unit_losses = self.ctx.get_cached_heuristics(curr_indices)
        lazy_pipes = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i])
        
        candidates = []
        up_limit = max(5, int(len(path_pipes) * 0.5))
        down_limit = max(10, self.ctx.num_pipes // 3)
        
        for up_pipe in path_pipes[-up_limit:]:
            u_idx = self.ctx.diam_to_idx[best_sol[up_pipe]] 
            if u_idx == len(self.ctx.diameters) - 1: continue
            
            for down_pipe in lazy_pipes[:down_limit]:
                if up_pipe == down_pipe: continue
                d_idx = self.ctx.diam_to_idx[best_sol[down_pipe]] 
                if d_idx == 0: continue
                
                test_sol = list(best_sol)
                test_sol[up_pipe] = self.ctx.diameters[u_idx + 1]
                test_sol[down_pipe] = self.ctx.diameters[d_idx - 1]
                
                t_idx = self.ctx.get_current_indices(test_sol)
                c, p, feas, _ = self.ctx.get_cached_stats(t_idx)
                
                if feas and p >= self.ctx.simulator.config.h_min:
                    surplus = p - self.ctx.simulator.config.h_min
                    score = c - (surplus * dyn_bonus) # ВИПРАВЛЕННЯ 6: Динамічний бонус
                    candidates.append((score, test_sol, {up_pipe}))

        for up_pipe in path_pipes[-int(up_limit*0.7):]:
            u_idx = self.ctx.diam_to_idx[best_sol[up_pipe]]
            if u_idx == len(self.ctx.diameters) - 1: continue
            for d1, d2 in itertools.combinations(lazy_pipes[:int(down_limit*0.7)], 2):
                if up_pipe in (d1, d2): continue
                d1_idx, d2_idx = self.ctx.diam_to_idx[best_sol[d1]], self.ctx.diam_to_idx[best_sol[d2]]
                if d1_idx == 0 or d2_idx == 0: continue
                
                test_sol = list(best_sol)
                test_sol[up_pipe] = self.ctx.diameters[u_idx + 1]
                test_sol[d1] = self.ctx.diameters[d1_idx - 1]
                test_sol[d2] = self.ctx.diameters[d2_idx - 1]
                
                t_idx = self.ctx.get_current_indices(test_sol)
                c, p, feas, _ = self.ctx.get_cached_stats(t_idx)
                if feas and p >= self.ctx.simulator.config.h_min:
                    surplus = p - self.ctx.simulator.config.h_min
                    score = c - (surplus * dyn_bonus)
                    candidates.append((score, test_sol, {up_pipe}))

        if not candidates: return best_sol
        candidates.sort(key=lambda x: x[0])
        best_score, best_candidate_sol, locked = candidates[0]
        
        squeezed = self.gradient_squeeze(best_candidate_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
        sq_idx = self.ctx.get_current_indices(squeezed)
        sq_c, _, _, _ = self.ctx.get_cached_stats(sq_idx)
        
        if sq_c < cost: return squeezed
        return best_sol

    def evaluate_candidate(self, base_solution, pipes_to_mod, mode, dyn_bonus):
        test_sol = list(base_solution)
        locked = set()
        valid = False
        for p_idx in pipes_to_mod:
            curr_d_idx = self.ctx.diam_to_idx[test_sol[p_idx]]
            if mode == "upgrade" and curr_d_idx < len(self.ctx.diameters) - 1:
                test_sol[p_idx] = self.ctx.diameters[curr_d_idx + 1]
                locked.add(p_idx)
                valid = True
            elif mode == "downgrade" and curr_d_idx > 0:
                test_sol[p_idx] = self.ctx.diameters[curr_d_idx - 1]
                valid = True
        
        if not valid: return float('inf'), -float('inf'), None
        
        squeezed_sol = self.gradient_squeeze(test_sol, locked_pipes=locked, max_passes=3, quick_mode=True, dyn_bonus=dyn_bonus)
        final_indices = self.ctx.get_current_indices(squeezed_sol)
        cost, p_min, _, _ = self.ctx.get_cached_stats(final_indices)
        
        p_surplus = p_min - self.ctx.simulator.config.h_min
        if p_surplus < 0: return float('inf'), -float('inf'), None
        
        score = cost - (p_surplus * dyn_bonus) # ВИПРАВЛЕННЯ 6
        return score, cost, squeezed_sol

# =============================================================================
# 4. KICK STRATEGIES
# =============================================================================
class KickStrategies:
    def __init__(self, ctx, local_search):
        self.ctx = ctx
        self.ls = local_search

    def forcing_hand_kick(self, solution):
        curr_indices = self.ctx.get_current_indices(solution)
        _, _, _, crit_node = self.ctx.get_cached_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return solution, set(), ""

        path_pipes, _ = self.ctx.get_dominant_path(curr_indices, crit_node)
        if not path_pipes: return solution, set(), ""
            
        kicked, locked = list(solution), set()
        for idx in path_pipes:
            curr_idx = curr_indices[idx] 
            if curr_idx < len(self.ctx.diameters) - 1:
                kicked[idx] = self.ctx.diameters[curr_idx + 1]
                locked.add(idx) 
        
        return kicked, locked, f"SHOCK: Upgraded Critical Path [Pipes: {len(path_pipes)} count]"

    def upstream_bottleneck_kick(self, solution):
        curr_indices = self.ctx.get_current_indices(solution)
        _, _, _, crit_node = self.ctx.get_cached_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        path_pipes, _ = self.ctx.get_dominant_path(curr_indices, crit_node)
        if not path_pipes: return None, None, ""
        
        best_candidate, reason = -1, ""
        for i in range(1, len(path_pipes)):
            prev_idx, curr_idx = path_pipes[i-1], path_pipes[i]
            d_prev, d_curr = curr_indices[prev_idx], curr_indices[curr_idx]
            if d_curr < d_prev and i < len(path_pipes) * 0.7:
                best_candidate, reason = curr_idx, f"Taper Violation (D{d_prev}->D{d_curr})"
                break
        
        if best_candidate == -1:
            unit_losses = self.ctx.get_cached_heuristics(curr_indices)
            max_loss = -1.0
            limit = max(1, len(path_pipes) // 2)
            for idx in path_pipes[:limit]:
                if unit_losses[idx] > max_loss:
                    max_loss, best_candidate, reason = unit_losses[idx], idx, f"Max Upstream Loss ({max_loss:.4f})"

        if best_candidate == -1: return None, None, ""
        
        kicked, locked = list(solution), set()
        curr_d_idx = curr_indices[best_candidate] 
        target_idx = min(len(self.ctx.diameters)-1, curr_d_idx + 2)
        if best_candidate in path_pipes[:2]: target_idx = max(target_idx, len(self.ctx.diameters)-2)

        if target_idx == curr_d_idx: return None, None, "" 

        kicked[best_candidate] = self.ctx.diameters[target_idx]
        locked.add(best_candidate)
        return kicked, locked, f"BOTTLENECK: Boosted Pipe {best_candidate} ({reason}) to {self.ctx.diameters[target_idx]}"

    def topological_inversion_kick(self, solution, tabu_set):
        curr_indices = self.ctx.get_current_indices(solution)
        _, _, _, crit_node = self.ctx.get_cached_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return None, None, "", None
        
        dominant_indices, dominant_path_nodes = self.ctx.get_dominant_path(curr_indices, crit_node)
        if not dominant_path_nodes: return None, None, "", None
        
        source = dominant_path_nodes[0]
        dominant_edges = set()
        dom_diameters = []
        for i, (u, v) in enumerate(zip(dominant_path_nodes[:-1], dominant_path_nodes[1:])):
            idx = dominant_indices[i]
            dominant_edges.add(tuple(sorted((u, v))))
            dom_diameters.append(curr_indices[idx])
            
        target_capacity_idx = int(np.mean(dom_diameters)) if dom_diameters else len(self.ctx.diameters)-1
        
        try:
            path_generator = nx.shortest_simple_paths(self.ctx.base_G_flow, source, crit_node)
            candidates = []
            start_time = time.time()
            # ВИПРАВЛЕННЯ 9: Timeout для генератора шляхів (макс 1 секунда)
            for path in path_generator:
                candidates.append(path)
                if len(candidates) >= 100 or (time.time() - start_time) > 1.0: break
        except: return None, None, "", None

        scored_paths = []
        for p_nodes in candidates:
            if p_nodes == dominant_path_nodes: continue
            p_indices, overlap, length = [], 0, 0
            for u, v in zip(p_nodes[:-1], p_nodes[1:]):
                edge_key = tuple(sorted((u, v)))
                if edge_key in dominant_edges: overlap += 1
                edge_data = self.ctx.simulator.graph.get_edge_data(u, v)
                if not edge_data: edge_data = self.ctx.simulator.graph.get_edge_data(v, u)
                if edge_data:
                    found_idx = -1
                    for key in edge_data:
                        if key in self.ctx.simulator.component_names:
                            found_idx = self.ctx.comp_name_to_idx[key]
                            break
                    if found_idx != -1:
                        p_indices.append(found_idx)
                        length += 1
            
            if length == 0: continue
            overlap_ratio = overlap / length
            signature = tuple(sorted(p_indices))
            if signature in tabu_set: continue
            scored_paths.append({'indices': p_indices, 'overlap': overlap_ratio, 'signature': signature})

        scored_paths.sort(key=lambda x: x['overlap'])
        if not scored_paths: return None, None, "All paths tabu", None

        best_alt = scored_paths[0]
        kicked, locked, push_count = list(solution), set(), 0
        for idx in best_alt['indices']:
            force_idx = max(target_capacity_idx, len(self.ctx.diameters)-2) 
            force_idx = min(force_idx, len(self.ctx.diameters)-1)
            curr = curr_indices[idx]
            if curr < force_idx:
                kicked[idx] = self.ctx.diameters[force_idx]
                locked.add(idx)
                push_count += 1
            elif curr == force_idx:
                 locked.add(idx)

        msg = f"Topo-Kick: Boosting Alt Path (Overlap {best_alt['overlap']:.2f}, {push_count} boosted)"
        return kicked, locked, msg, best_alt['signature']

    def micro_trim_kick(self, solution):
        curr_indices, periphery_pipes = self.ctx.get_lazy_periphery(solution)
        if not periphery_pipes: return None, None, ""
        
        kicked, locked, shrunk_count = list(solution), set(), 0
        for idx in periphery_pipes:
            curr_d_idx = curr_indices[idx]
            if curr_d_idx > 0: 
                kicked[idx] = self.ctx.diameters[curr_d_idx - 1]
                locked.add(idx)
                shrunk_count += 1
            if shrunk_count >= 5: break
                
        if shrunk_count > 0:
            healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
            if is_feasible:
                return healed_sol, locked, f"FINISHER: Force-shrunk {shrunk_count} periphery pipes. Healed {boosts}x."
        return None, None, ""
    
    def sync_trim_kick(self, solution, dyn_bonus):
        curr_indices, periphery_pipes = self.ctx.get_lazy_periphery(solution)
        if not periphery_pipes: return None, None, ""
        candidates = []
        
        # ВИПРАВЛЕННЯ 8: Захист від комбінаторного вибуху на великих мережах
        combo_limit = min(30, max(10, self.ctx.num_pipes // 3))
        all_pairs = list(itertools.combinations(periphery_pipes[:combo_limit], 2))
        random.shuffle(all_pairs)
        
        for d1, d2 in all_pairs[:60]: # Оцінюємо максимум 60 випадкових пар!
            d1_idx, d2_idx = curr_indices[d1], curr_indices[d2]
            if d1_idx == 0 or d2_idx == 0: continue
            
            test_sol = list(solution)
            test_sol[d1] = self.ctx.diameters[d1_idx - 1]
            test_sol[d2] = self.ctx.diameters[d2_idx - 1]
            locked = {d1, d2}
            
            healed_sol, is_feasible, boosts = self.ls.heal_network(test_sol, locked)
            if is_feasible:
                h_idx = self.ctx.get_current_indices(healed_sol)
                h_c, h_p, _, _ = self.ctx.get_cached_stats(h_idx) 
                h_surplus = h_p - self.ctx.simulator.config.h_min
                h_score = h_c - (h_surplus * dyn_bonus)
                candidates.append((h_score, healed_sol, locked, boosts))
                
        if not candidates: return None, None, ""
        candidates.sort(key=lambda x: x[0])
        validated_candidates = []
        
        for _, healed_sol, locked, boosts in candidates[:5]:
            squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
            sq_idx = self.ctx.get_current_indices(squeezed)
            sq_c, _, _, _ = self.ctx.get_cached_stats(sq_idx)
            validated_candidates.append((sq_c, healed_sol, locked, boosts))
                
        if not validated_candidates: return None, None, ""
        validated_candidates.sort(key=lambda x: x[0])
        chosen = random.choice(validated_candidates[:3])
        _, best_final_sol, best_locked, best_boosts = chosen
        
        return best_final_sol, best_locked, f"SYNC-TRIM: Shrunk Pipes {[p+1 for p in best_locked]}. Healed {best_boosts}x."

    def loop_balancing_kick(self, solution, dyn_bonus):
        curr_indices = self.ctx.get_current_indices(solution)
        _, _, _, crit_node = self.ctx.get_cached_stats(curr_indices)
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
                curr_d_idx = curr_indices[candidate_idx]
                if curr_d_idx < 2: continue 
                
                max_drop = min(3, curr_d_idx) 
                for drop in range(max_drop, 0, -1):
                    if drop < best_drop_achieved: continue
                        
                    target_d_idx = curr_d_idx - drop
                    kicked, locked = list(solution), set()
                    kicked[candidate_idx] = self.ctx.diameters[target_d_idx]
                    locked.add(candidate_idx)
                    
                    healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
                    if is_feasible:
                        test_squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=4, quick_mode=True, dyn_bonus=dyn_bonus)
                        sq_indices = self.ctx.get_current_indices(test_squeezed)
                        sq_cost, _, _, _ = self.ctx.get_cached_stats(sq_indices)
                        msg = f"FLOW STEER: Cut Pipe {candidate_idx + 1} (-{drop}). Healed {boosts}x."
                        candidates.append((sq_cost, healed_sol, locked, msg))

        if not candidates: return None, None, "FLOW STEER: Exhaustive search found no valid bypass."
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])
        return chosen[1], chosen[2], chosen[3]

# =============================================================================
# 5. ORCHESTRATOR
# =============================================================================
class AnalyticalSolver:
    def __init__(self, simulator, available_diameters, v_opt=1.0, max_iters=10, time_limit_sec=900):
        self.ctx = SolverContext(simulator, available_diameters, v_opt)
        self.pool = SolutionPool(self.ctx)
        self.ls = LocalSearch(self.ctx)
        self.kicker = KickStrategies(self.ctx, self.ls)
        
        self.max_iters = max_iters
        self.time_limit_sec = time_limit_sec 
        self.best_found_solution = None
        
        self.BEAM_WIDTH = max(5, self.ctx.num_pipes // 10)
        self.SINGLE_CANDIDATES = max(6, self.ctx.num_pipes // 5)
        
        # ВИПРАВЛЕННЯ 3: Epsilon-Greedy Bandit для стратегій
        self.strategies = ["TOPO-DIV", "LOOP_BALANCE", "SYNC_TRIM", "FINISHER", "BOTTLENECK", "SHOCK"]
        self.strat_wins = {s: 1.0 for s in self.strategies}
        self.strat_tries = {s: 1.0 for s in self.strategies}

    def _calculate_ideal_d(self, flow, target_v):
        if abs(flow) < 1e-6: return self.ctx.diameters[0]
        d_ideal = math.sqrt((4.0 * abs(flow)) / (math.pi * target_v))
        pos = bisect.bisect_left(self.ctx.diameters, d_ideal)
        if pos < len(self.ctx.diameters): return self.ctx.diameters[pos]
        return self.ctx.diameters[-1]

    def _make_diverse_seeds(self):
        seeds = []
        velocities_to_try = [1.0, 1.2, 0.8] 
        for v in velocities_to_try:
            sol = [self.ctx.diameters[-1]] * self.ctx.num_pipes
            for _ in range(self.max_iters):
                flows, _ = self.ctx.simulator.get_hydraulic_state(sol)
                new = [self._calculate_ideal_d(q, v) for q in flows] 
                if new == sol: break
                sol = new
                
            idx = self.ctx.get_current_indices(sol)
            _, p, feas, _ = self.ctx.get_cached_stats(idx)
            if not feas or p < self.ctx.simulator.config.h_min:
                sol, is_healed, _ = self.ls.heal_network(sol, set())
                if not is_healed: continue 
                    
            squeezed_sol = self.ls.gradient_squeeze(sol, quick_mode=True)
            seeds.append(squeezed_sol)
        return seeds

    def solve_standalone(self):
        start_time = time.time() 
        self.pool.clear_all()
        print("[AnalyticalSolver] 1. Generating Diverse Seeds...")
        seeds = self._make_diverse_seeds()
        
        # Ініціалізація динамічного бонусу
        best_initial_cost = float('inf')
        for s in seeds:
            idx = self.ctx.get_current_indices(s)
            c, _, _, _ = self.ctx.get_cached_stats(idx)
            if c < best_initial_cost: best_initial_cost = c
            
        dyn_bonus = best_initial_cost * 0.001 
        
        for s in seeds:
            idx = self.ctx.get_current_indices(s)
            c, p, _, _ = self.ctx.get_cached_stats(idx)
            p_surplus = p - self.ctx.simulator.config.h_min
            score = c - (p_surplus * dyn_bonus)
            self.pool.active_pool.append((score, c, s))
            self.pool.add_to_tabu(s, c)
            
        global_best_cost = min(x[1] for x in self.pool.active_pool)
        global_best_sol = next(x[2] for x in self.pool.active_pool if x[1] == global_best_cost)
        print(f"   > Seeds initialized. Best Start: {global_best_cost/1e6:.4f}M$")

        MAX_ROUNDS = 50 
        stagnation_counter = 0

        for round_idx in range(MAX_ROUNDS):
            if time.time() - start_time > self.time_limit_sec:
                self.ctx.log(f"[INFO] Time limit ({self.time_limit_sec}s) reached! Stopping.")
                break
                
            # ВИПРАВЛЕННЯ 6: Перерахунок динамічного бонусу (0.1% від рекорду за кожен метр)
            dyn_bonus = global_best_cost * 0.001
                
            if round_idx > 0 and round_idx % 6 == 0:
                self.pool.kick_tabu_set.clear()
            
            next_gen = []
            
            # SWAP SEARCH 
            swapped = self.ls.swap_search(global_best_sol, dyn_bonus)
            idx = self.ctx.get_current_indices(swapped)
            c, p, _, _ = self.ctx.get_cached_stats(idx)
            if c < global_best_cost:
                 diff = global_best_cost - c
                 global_best_cost = c
                 global_best_sol = swapped
                 print(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({global_best_cost/1e6:.4f}M$)")
                 stagnation_counter = 0
                 self.pool.kick_tabu_set.clear()
                 p_surplus = p - self.ctx.simulator.config.h_min
                 score = c - (p_surplus * dyn_bonus)
                 self.pool.active_pool.insert(0, (score - (global_best_cost * 0.1), c, swapped))

            # FORCE STRATEGY CYCLE (Epsilon-Greedy Bandit)
            if stagnation_counter >= 1 or round_idx % 2 == 0:
                # З імовірністю 20% обираємо випадково, інакше - найкращу за Win Rate
                if random.random() < 0.2:
                    strategy = random.choice(self.strategies)
                else:
                    strategy = max(self.strategies, key=lambda s: self.strat_wins[s] / self.strat_tries[s])
                
                self.strat_tries[strategy] += 1
                self.ctx.log(f"[FORCE] Applying '{strategy}' (WinRate: {self.strat_wins[strategy]/self.strat_tries[strategy]:.2f})...")
                
                pool_sols = [x[2] for x in self.pool.active_pool[:3]]
                if global_best_sol not in pool_sols: pool_sols.append(global_best_sol)
                kick_target = random.choice(pool_sols)
                
                forced_sol, locked, path_sig = None, None, None
                
                if strategy == "SHOCK": forced_sol, locked, log_msg = self.kicker.forcing_hand_kick(kick_target)
                elif strategy == "BOTTLENECK": forced_sol, locked, log_msg = self.kicker.upstream_bottleneck_kick(kick_target)
                elif strategy == "LOOP_BALANCE": forced_sol, locked, log_msg = self.kicker.loop_balancing_kick(kick_target, dyn_bonus)
                elif strategy == "FINISHER": forced_sol, locked, log_msg = self.kicker.micro_trim_kick(kick_target)
                elif strategy == "SYNC_TRIM": forced_sol, locked, log_msg = self.kicker.sync_trim_kick(kick_target, dyn_bonus)
                else: forced_sol, locked, log_msg, path_sig = self.kicker.topological_inversion_kick(kick_target, self.pool.kick_tabu_set)
                
                if forced_sol and locked:
                    self.ctx.log(f"     -> {log_msg}")
                    starved_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=8, dyn_bonus=dyn_bonus)
                    final_sol = self.ls.gradient_squeeze(starved_sol, dyn_bonus=dyn_bonus)
                    
                    idx = self.ctx.get_current_indices(final_sol)
                    c, p, _, _ = self.ctx.get_cached_stats(idx)
                    
                    if p >= self.ctx.simulator.config.h_min:
                        p_surplus = p - self.ctx.simulator.config.h_min
                        score = c - (p_surplus * dyn_bonus)
                        
                        if stagnation_counter >= 3:
                            self.ctx.log("     [INFO] Flushing active pool to force exploration!")
                            self.pool.active_pool.clear()
                            stagnation_counter = 0 # Лікуємо амнезію пулу
                            
                        self.pool.active_pool.insert(0, (score, c, final_sol))
                        
                        if c < global_best_cost:
                            diff = global_best_cost - c
                            global_best_cost = c
                            global_best_sol = final_sol
                            self.strat_wins[strategy] += 5.0 # Велика винагорода за рекорд
                            self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({global_best_cost/1e6:.4f}M$)")
                        else:
                            self.strat_wins[strategy] += 0.5 # Мала винагорода за життєздатність
                            
                        self.ctx.log(f"     -> Injection Successful: {c/1e6:.4f}M$")
                        if path_sig: self.pool.kick_tabu_set.add(path_sig)
                    else:
                        self.ctx.log(f"     -> Injection Failed (Infeasible)")
                        if path_sig: self.pool.kick_tabu_set.add(path_sig)
                else:
                    self.ctx.log(f"     -> No valid kick generated.")

            # BEAM SEARCH EXPANSION
            for _, _, parent_sol in self.pool.active_pool:
                curr_indices = self.ctx.get_current_indices(parent_sol)
                unit_losses = self.ctx.get_cached_heuristics(curr_indices)
                
                high_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i], reverse=True)
                low_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i]) 
                
                for pipe_idx in high_friction[:self.SINGLE_CANDIDATES]:
                    s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "upgrade", dyn_bonus)
                    if sol: next_gen.append((s, c, sol))
                
                for pipe_idx in low_friction[:4]:
                     s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "downgrade", dyn_bonus)
                     if sol: next_gen.append((s, c, sol))

                if len(high_friction) >= 2:
                    for p1, p2 in itertools.combinations(high_friction[:4], 2):
                        s, c, sol = self.ls.evaluate_candidate(parent_sol, [p1, p2], "upgrade", dyn_bonus)
                        if sol: next_gen.append((s, c, sol))
            
            self.ctx.log(f"   > [R{round_idx+1}] Beam Search completed. Pool expanded.")
            
            if not next_gen:
                stagnation_counter += 1
                continue

            next_gen.sort(key=lambda x: x[0]) 
            unique_next_pool = []
            found_new_record = False
            
            for rank, (score, cost, sol) in enumerate(next_gen):
                if self.pool.is_tabu(sol, cost): continue
                # ВИПРАВЛЕННЯ 5: Фільтр різноманіття (відстань Хемінга >= 2)
                if not self.pool.is_diverse_enough(sol, min_distance=2): continue
                
                if len(unique_next_pool) < self.BEAM_WIDTH:
                    if rank == 0: refined_sol = self.ls.gradient_squeeze(sol, dyn_bonus=dyn_bonus)
                    elif rank < 3: refined_sol = self.ls.gradient_squeeze(sol, max_passes=4, quick_mode=True, dyn_bonus=dyn_bonus)
                    else: refined_sol = sol 

                    idx_sol = self.ctx.get_current_indices(refined_sol)
                    real_cost, p_min, _, _ = self.ctx.get_cached_stats(idx_sol)
                    
                    if p_min >= self.ctx.simulator.config.h_min:
                        real_p_surplus = p_min - self.ctx.simulator.config.h_min
                        real_score = real_cost - (real_p_surplus * dyn_bonus)
                        unique_next_pool.append((real_score, real_cost, refined_sol))
                        self.pool.add_to_tabu(refined_sol, real_cost)
                        
                        if real_cost < global_best_cost:
                            diff = global_best_cost - real_cost
                            global_best_cost = real_cost
                            global_best_sol = refined_sol
                            found_new_record = True
                            print(f"   > [R{round_idx+1}] 💎 New Record: -${diff:,.0f} ({global_best_cost/1e6:.4f}M$)")

            if found_new_record:
                stagnation_counter = 0
                self.pool.kick_tabu_set.clear()
            else:
                stagnation_counter += 1
                if stagnation_counter % 8 == 0:
                    self.pool.kick_tabu_set.clear()

            if unique_next_pool:
                unique_next_pool.sort(key=lambda x: x[0])
                self.pool.active_pool = unique_next_pool[:self.BEAM_WIDTH]

        total_time = time.time() - start_time
        print(f"[AnalyticalSolver] Фінал: {global_best_cost/1e6:.4f}M$ (Time: {total_time/60:.1f}m)")
        self.best_found_solution = global_best_sol 
        return global_best_sol