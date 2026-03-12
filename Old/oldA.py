import math
import random
import sys
import numpy as np
import itertools
import networkx as nx
from collections import deque

class AnalyticalSolver:
    def __init__(self, simulator, available_diameters, v_opt=1.0, max_iters=10):
        self.simulator = simulator
        self.diameters = sorted(available_diameters)
        self.v_opt = v_opt
        self.max_iters = max_iters
        self.num_pipes = simulator.n_variables
        
        self._tabu_fingerprints = set()

        costs = self.simulator.costs 
        avg_cost_diff = 0
        count = 0
        if len(costs) > 1:
            for c1, c2 in zip(costs[:-1], costs[1:]):
                avg_cost_diff += abs(c1 - c2)
                count += 1
            avg_cost_diff = (avg_cost_diff / count) if count > 0 else 50
        else:
            avg_cost_diff = 50

        avg_length = sum(self.simulator.lengths) / len(self.simulator.lengths)
        self.adaptive_bonus = avg_cost_diff * (avg_length * 0.8) 
        
        print(f"[AnalyticalSolver] Pressure Valuation: ${self.adaptive_bonus:,.0f} per meter")

    def _log(self, message):
        print(f"   {message}")

    def _get_fingerprint(self, solution, known_cost=None):
        sol_tuple = tuple(solution)
        if known_cost is not None:
            cost_bucket = int(known_cost / 50) * 50 
            return (cost_bucket, sol_tuple)
        idx = [self.diameters.index(d) if d in self.diameters else 0 for d in solution]
        cost, _, _, _ = self.simulator.get_stats(idx)
        return (int(cost / 50) * 50, sol_tuple)

    def _indices_to_path_str(self, pipe_indices):
        if not pipe_indices: return "[]"
        return f"[Pipes: {len(pipe_indices)} count]"

    def _calculate_ideal_d(self, flow):
        if abs(flow) < 1e-6: return self.diameters[0]
        d_ideal = math.sqrt((4.0 * abs(flow)) / (math.pi * self.v_opt))
        valid_ds = [d for d in self.diameters if d >= d_ideal]
        return valid_ds[0] if valid_ds else self.diameters[-1]

    def _get_current_indices(self, solution):
        return [self.diameters.index(d) if d in self.diameters else len(self.diameters)-1 for d in solution]

    def _gradient_squeeze(self, solution, locked_pipes=None, max_passes=None, quick_mode=False, verbose=False):
        """
        COMPENSATORY SQUEEZE:
        If reducing a pipe fails due to pressure violation, tries to 'buy' validity
        by expanding a cheaper downstream bottleneck pipe.
        """
        if locked_pipes is None: locked_pipes = set()
        current_solution = list(solution)
        improved = True
        
        lengths = self.simulator.lengths
        costs_array = self.simulator.costs 
        passes = 0
        
        active_indices = [i for i in range(self.num_pipes) if i not in locked_pipes]
        
        # Initial Stats
        curr_indices = self._get_current_indices(current_solution)
        cost, p_min, _, crit_node = self.simulator.get_stats(curr_indices)
        
        if verbose:
            surplus = p_min - self.simulator.config.h_min
            self._log(f"   [SQUEEZE] Start: {cost/1e6:.4f}M. Headroom: {surplus:.2f}m")

        while improved:
            improved = False
            passes += 1
            if max_passes and passes > max_passes: break
            
            curr_indices = self._get_current_indices(current_solution)
            unit_losses = self.simulator.get_heuristics(curr_indices)
            
            # Identify Candidates
            savings_candidates = []
            current_headroom = p_min - self.simulator.config.h_min
            
            for i in active_indices:
                if quick_mode and unit_losses[i] > 0.05: continue
                
                current_idx = curr_indices[i]
                if current_idx > 0:
                    dollar_save = lengths[i] * (costs_array[current_idx] - costs_array[current_idx - 1])
                    if dollar_save < 50.0: continue
                    
                    # Smart Reserve Logic
                    if dollar_save < 40000 and current_headroom < 1.5: continue

                    risk = unit_losses[i] + 1e-9
                    score = dollar_save / risk
                    savings_candidates.append((i, score, dollar_save))
            
            if not savings_candidates: break

            savings_candidates.sort(key=lambda x: x[1], reverse=True)
            limit = 5 if quick_mode else 20
            candidates_to_try = savings_candidates[:limit]

            batch_count = 0
            max_batch = 5 
            
            for idx, _, save in candidates_to_try:
                current_idx = self.diameters.index(current_solution[idx])
                
                # 1. Try Standard Reduction
                test_solution = list(current_solution)
                test_solution[idx] = self.diameters[current_idx - 1]
                
                t_indices = self._get_current_indices(test_solution)
                t_cost, t_p, t_feas, t_crit = self.simulator.get_stats(t_indices)
                
                if t_feas and t_p >= self.simulator.config.h_min:
                    # Success!
                    current_solution = test_solution
                    p_min = t_p
                    improved = True
                    batch_count += 1
                    if verbose:
                        surplus = t_p - self.simulator.config.h_min
                        self._log(f"      -> Reduced Pipe {idx}. Saved ${save:,.0f}. Headroom: {surplus:.2f}m")
                    if batch_count >= max_batch: break
                
                else:
                    # --- COMPENSATORY LOGIC (Plan B) ---
                    # Reduction failed. Can we fix it by expanding a cheap neighbor?
                    # Only try this for significant savings (> $20k) and NOT in quick_mode
                    if not quick_mode and save > 20000 and t_crit is not None:
                        
                        # Find pipes connected to the Critical Node (where pressure failed)
                        neighbors = []
                        for u, v, k in self.simulator.graph.edges(keys=True):
                            if (u == t_crit or v == t_crit) and k in self.simulator.component_names:
                                n_idx = self.simulator.component_names.index(k)
                                if n_idx != idx and n_idx not in locked_pipes: # Don't expand the pipe we just shrunk
                                    neighbors.append(n_idx)
                        
                        best_fix = None
                        best_net_save = 0
                        
                        # Try expanding each neighbor
                        for n_idx in neighbors:
                            n_curr_d = self.diameters.index(test_solution[n_idx])
                            if n_curr_d < len(self.diameters) - 1:
                                fix_solution = list(test_solution)
                                fix_solution[n_idx] = self.diameters[n_curr_d + 1] # Expand neighbor
                                
                                # Calculate Cost of Expansion
                                expansion_cost = self.simulator.lengths[n_idx] * (
                                    self.simulator.costs[n_curr_d + 1] - self.simulator.costs[n_curr_d]
                                )
                                
                                net_save = save - expansion_cost
                                
                                # Only proceed if we still save money
                                if net_save > 5000: 
                                    f_indices = self._get_current_indices(fix_solution)
                                    _, f_p, f_feas, _ = self.simulator.get_stats(f_indices)
                                    
                                    if f_feas and f_p >= self.simulator.config.h_min:
                                        if net_save > best_net_save:
                                            best_net_save = net_save
                                            best_fix = fix_solution
                                            best_neighbor = n_idx

                        if best_fix:
                            current_solution = best_fix
                            p_min = f_p # Approx
                            improved = True
                            batch_count += 1
                            if verbose:
                                self._log(f"      -> [COMBO] Reduced {idx} & Expanded {best_neighbor}. Net Save: ${best_net_save:,.0f}")
                            if batch_count >= max_batch: break

        return current_solution
    
    def _make_diverse_seeds(self):
        seeds = []
        sol1 = [self.diameters[-1]] * self.num_pipes
        for _ in range(self.max_iters):
            flows, _ = self.simulator.get_hydraulic_state(sol1)
            new = [self._calculate_ideal_d(q) for q in flows]
            if new == sol1: break
            sol1 = new
        seeds.append(self._gradient_squeeze(sol1))
        
        sol2 = [self.diameters[-1]] * self.num_pipes
        seeds.append(self._gradient_squeeze(sol2, quick_mode=True))
        
        return seeds

    def _forcing_hand_kick(self, solution):
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self.simulator.get_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return solution, set(), ""

        path_pipes, _ = self._get_dominant_path(curr_indices, crit_node)
        if not path_pipes: return solution, set(), ""
            
        kicked = list(solution)
        locked = set()
        
        for idx in path_pipes:
            curr_idx = self.diameters.index(kicked[idx])
            if curr_idx < len(self.diameters) - 1:
                kicked[idx] = self.diameters[curr_idx + 1]
                locked.add(idx) 
        
        log_msg = f"SHOCK: Upgraded Critical Path {self._indices_to_path_str(path_pipes)}"
        return kicked, locked, log_msg

    def _upstream_bottleneck_kick(self, solution):
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self.simulator.get_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        path_pipes, _ = self._get_dominant_path(curr_indices, crit_node)
        if not path_pipes: return None, None, ""
        
        best_candidate = -1
        reason = ""
        
        for i in range(1, len(path_pipes)):
            prev_idx = path_pipes[i-1]
            curr_idx = path_pipes[i]
            d_prev = curr_indices[prev_idx]
            d_curr = curr_indices[curr_idx]
            
            if d_curr < d_prev and i < len(path_pipes) * 0.7:
                best_candidate = curr_idx
                reason = f"Taper Violation (D{d_prev}->D{d_curr})"
                break
        
        if best_candidate == -1:
            unit_losses = self.simulator.get_heuristics(curr_indices)
            max_loss = -1.0
            limit = max(1, len(path_pipes) // 2)
            for idx in path_pipes[:limit]:
                if unit_losses[idx] > max_loss:
                    max_loss = unit_losses[idx]
                    best_candidate = idx
                    reason = f"Max Upstream Loss ({max_loss:.4f})"

        if best_candidate == -1: return None, None, ""
        
        kicked = list(solution)
        locked = set()
        curr_d_idx = self.diameters.index(kicked[best_candidate])
        target_idx = min(len(self.diameters)-1, curr_d_idx + 2)
        if best_candidate in path_pipes[:2]: target_idx = max(target_idx, len(self.diameters)-2)

        kicked[best_candidate] = self.diameters[target_idx]
        locked.add(best_candidate)
        
        return kicked, locked, f"BOTTLENECK: Boosted Pipe {best_candidate} ({reason}) to {self.diameters[target_idx]}"

    def _topological_inversion_kick(self, solution, tabu_set=None):
        """
        TOPOLOGICAL DIVERSITY SEARCH (Relaxed):
        If strict alternatives are not found, allows paths with higher overlap.
        """
        if tabu_set is None: tabu_set = set()
        # ... (початок методу без змін до 'candidates') ...
        # (Весь код пошуку dominant_path залишається)
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self.simulator.get_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return None, None, "", None
        source = self.simulator.sources[0]
        G_flow = nx.Graph()
        for u, v, k in self.simulator.graph.edges(keys=True):
            if k in self.simulator.component_names:
                idx = self.simulator.component_names.index(k)
                weight = 100.0 / (curr_indices[idx] + 1)
                G_flow.add_edge(u, v, weight=weight, id=idx)
        try:
            dominant_path_nodes = nx.shortest_path(G_flow, source, crit_node, weight='weight')
        except: return None, None, "", None
        dominant_edges = set()
        dominant_indices = []
        dom_diameters = []
        for u, v in zip(dominant_path_nodes[:-1], dominant_path_nodes[1:]):
            edge_data = G_flow.get_edge_data(u, v)
            if edge_data is None: edge_data = G_flow.get_edge_data(v, u)
            if edge_data:
                idx = edge_data['id'] if 'id' in edge_data else list(edge_data.values())[0]['id']
                dominant_indices.append(idx)
                dominant_edges.add(tuple(sorted((u, v))))
                dom_diameters.append(curr_indices[idx])
        target_capacity_idx = int(np.mean(dom_diameters)) if dom_diameters else len(self.diameters)-1
        
        G_simple = nx.Graph()
        for u, v in self.simulator.graph.edges(): G_simple.add_edge(u, v)
        
        candidates = []
        try:
            path_generator = nx.shortest_simple_paths(G_simple, source, crit_node)
            candidates = list(itertools.islice(path_generator, 100)) # Більше кандидатів
        except: return None, None, "", None

        scored_paths = []
        for p_nodes in candidates:
            if p_nodes == dominant_path_nodes: continue
            
            p_indices = []
            overlap = 0
            length = 0
            for u, v in zip(p_nodes[:-1], p_nodes[1:]):
                edge_key = tuple(sorted((u, v)))
                if edge_key in dominant_edges: overlap += 1
                edge_data = self.simulator.graph.get_edge_data(u, v)
                if edge_data is None: edge_data = self.simulator.graph.get_edge_data(v, u)
                if edge_data:
                    found_idx = -1
                    for key in edge_data:
                        if key in self.simulator.component_names:
                            found_idx = self.simulator.component_names.index(key)
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
        
        # RELAXED FALLBACK: Якщо немає ідеальних, беремо будь-який не з Табу
        if not scored_paths: 
             return None, None, "All paths tabu", None

        best_alt = scored_paths[0]
        
        # ... (Код виконання PUSH залишається без змін) ...
        target_indices = best_alt['indices']
        kicked = list(solution)
        locked = set()
        push_count = 0
        for idx in target_indices:
            force_idx = max(target_capacity_idx, len(self.diameters)-2) 
            force_idx = min(force_idx, len(self.diameters)-1)
            curr = curr_indices[idx]
            if curr < force_idx:
                kicked[idx] = self.diameters[force_idx]
                locked.add(idx)
                push_count += 1
            elif curr == force_idx:
                 locked.add(idx)

        msg = f"Topo-Kick: Boosting Alt Path (Overlap {best_alt['overlap']:.2f}, {push_count} boosted)"
        return kicked, locked, msg, best_alt['signature']

    def _swap_search(self, solution):
        curr_indices = self._get_current_indices(solution)
        unit_losses = self.simulator.get_heuristics(curr_indices)
        best_sol = list(solution)
        high_f = sorted(range(self.num_pipes), key=lambda i: unit_losses[i], reverse=True)
        low_f  = sorted(range(self.num_pipes), key=lambda i: unit_losses[i])
        
        # Check Top 20
        for h_pipe in high_f[:20]:
            for l_pipe in low_f[:20]:
                if h_pipe == l_pipe: continue
                d_high = best_sol[h_pipe]
                d_low  = best_sol[l_pipe]
                h_idx = self.diameters.index(d_high)
                l_idx = self.diameters.index(d_low)
                
                if h_idx < len(self.diameters)-1 and l_idx > 0:
                    test_sol = list(best_sol)
                    test_sol[h_pipe] = self.diameters[h_idx + 1] 
                    test_sol[l_pipe] = self.diameters[l_idx - 1]
                    
                    # Cost Analysis for Relaxed Check
                    len_h = self.simulator.lengths[h_pipe]
                    len_l = self.simulator.lengths[l_pipe]
                    cost_h_up = self.simulator.costs[h_idx+1] - self.simulator.costs[h_idx]
                    cost_l_down = self.simulator.costs[l_idx] - self.simulator.costs[l_idx-1]
                    
                    direct_cost_diff = (len_h * cost_h_up) - (len_l * cost_l_down)
                    
                    # RELAXED THRESHOLD: Allow swap even if it costs up to $50,000 upfront
                    # This allows "investing" in main pipes
                    if direct_cost_diff > 50000: continue 

                    squeezed = self._gradient_squeeze(test_sol, locked_pipes={h_pipe}, max_passes=2, quick_mode=True)
                    idx = self._get_current_indices(squeezed)
                    cost, p_min, _, _ = self.simulator.get_stats(idx)
                    
                    curr_idx = self._get_current_indices(best_sol)
                    curr_cost, _, _, _ = self.simulator.get_stats(curr_idx)
                    
                    if p_min >= self.simulator.config.h_min and cost < curr_cost:
                         diff = curr_cost - cost
                         self._log(f"[SWAP] {h_pipe} <-> {l_pipe}: -${diff:,.0f}")
                         return squeezed
        return best_sol
    
    def _get_dominant_path(self, curr_indices, crit_node):
        source = self.simulator.sources[0]
        G_flow = nx.Graph()
        for u, v, k in self.simulator.graph.edges(keys=True):
            if k in self.simulator.component_names:
                idx = self.simulator.component_names.index(k)
                # Вода тече туди, де ширше (вага обернена діаметру)
                weight = 100.0 / (curr_indices[idx] + 1)
                G_flow.add_edge(u, v, weight=weight, id=idx)
        
        try:
            path_nodes = nx.shortest_path(G_flow, source, crit_node, weight='weight')
            path_pipes = []
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = G_flow.get_edge_data(u, v)
                if edge_data and 'id' in edge_data:
                    path_pipes.append(edge_data['id'])
            return path_pipes, path_nodes
        except:
            return [], []
    
    def _loop_balancing_kick(self, solution):
        """
        FLOW STEERING (Pure Adaptive Grid Search):
        Honest search: drops the lazy giant by 1..N sizes, 
        and boosts the bypass. Only locks the Giant, allowing Squeeze 
        to perfectly trim the bloated bypass.
        """
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self.simulator.get_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        G_simple = nx.Graph()
        for u, v in self.simulator.graph.edges(): G_simple.add_edge(u, v)
        try: cycles = nx.cycle_basis(G_simple)
        except: return None, None, ""
        if not cycles: return None, None, "No cycles found"

        unit_losses = self.simulator.get_heuristics(curr_indices)
        best_candidate = -1
        best_cycle = []
        max_score = -1.0

        for cycle_nodes in cycles:
            cycle_indices = []
            full_cycle = cycle_nodes + [cycle_nodes[0]]
            for u, v in zip(full_cycle[:-1], full_cycle[1:]):
                edge_data = self.simulator.graph.get_edge_data(u, v)
                if edge_data is None: edge_data = self.simulator.graph.get_edge_data(v, u)
                if edge_data:
                    idx = -1
                    for key in edge_data:
                        if key in self.simulator.component_names:
                            idx = self.simulator.component_names.index(key)
                            break
                    if idx != -1: cycle_indices.append(idx)

            for idx in cycle_indices:
                d_idx = curr_indices[idx]
                loss = unit_losses[idx]
                cost = self.simulator.costs[d_idx]
                if d_idx > 2: 
                    score = cost / (loss + 1e-9)
                    if score > max_score:
                        max_score = score
                        best_candidate = idx
                        best_cycle = cycle_indices

        if best_candidate == -1: return None, None, "No lazy giants found"

        curr_d_idx = curr_indices[best_candidate]
        
        best_combo_sol = None
        best_combo_locked = set()
        best_combo_score = float('inf')
        best_msg = ""

        # ADAPTIVE GRID SEARCH
        max_drop = min(4, curr_d_idx)
        
        for drop in range(1, max_drop + 1):
            target_d_idx = curr_d_idx - drop
            
            for boost in range(0, 4):
                kicked = list(solution)
                locked = set()
                
                # A. Відрізаємо гіганта
                kicked[best_candidate] = self.diameters[target_d_idx]
                locked.add(best_candidate) # <--- БЛОКУЄМО ТІЛЬКИ ГІГАНТА
                
                # B. Накачуємо обхід (без блокування!)
                for idx in best_cycle:
                    if idx == best_candidate: continue
                    
                    b_curr_d = curr_indices[idx]
                    b_target = min(b_curr_d + boost, len(self.diameters) - 1)
                    
                    if b_curr_d < b_target:
                        kicked[idx] = self.diameters[b_target]
                        # ТУТ БІЛЬШЕ НЕМАЄ locked.add(idx) - Squeeze зможе їх зрізати!

                # C. Перевірка
                test_indices = self._get_current_indices(kicked)
                _, t_p, t_feas, _ = self.simulator.get_stats(test_indices)
                
                if t_feas and t_p >= self.simulator.config.h_min:
                    # Оскільки обхід не заблокований, Squeeze ідеально зріже зайвий жир
                    # Збільшуємо max_passes до 4, щоб він встиг прибрати всі надлишки для чесної оцінки
                    test_squeezed = self._gradient_squeeze(kicked, locked_pipes=locked, max_passes=4, quick_mode=True)
                    sq_indices = self._get_current_indices(test_squeezed)
                    sq_cost, sq_p, _, _ = self.simulator.get_stats(sq_indices)
                    
                    # Оцінюємо по фінальній (оптимізованій) вартості
                    score = sq_cost 
                    
                    if score < best_combo_score:
                        best_combo_score = score
                        best_combo_sol = kicked 
                        best_combo_locked = locked 
                        best_msg = f"FLOW STEER: Giant -{drop} sizes, Bypass +{boost} sizes. (Est. Cost: {score/1e6:.3f}M)"
                        break # Знайшли найдешевший boost для цього drop, йдемо до наступного drop

        if best_combo_sol:
             return best_combo_sol, best_combo_locked, best_msg
             
        return None, None, "FLOW STEER: Failed to find feasible drop/boost combination."

    def _evaluate_candidate(self, base_solution, pipes_to_mod, mode="upgrade"):
        test_sol = list(base_solution)
        locked = set()
        valid = False
        for p_idx in pipes_to_mod:
            curr_d_idx = self.diameters.index(test_sol[p_idx])
            if mode == "upgrade":
                if curr_d_idx < len(self.diameters) - 1:
                    test_sol[p_idx] = self.diameters[curr_d_idx + 1]
                    locked.add(p_idx)
                    valid = True
            elif mode == "downgrade":
                if curr_d_idx > 0:
                    test_sol[p_idx] = self.diameters[curr_d_idx - 1]
                    valid = True
        
        if not valid: return float('inf'), -float('inf'), None
        squeezed_sol = self._gradient_squeeze(test_sol, locked_pipes=locked, max_passes=3, quick_mode=True)
        final_indices = self._get_current_indices(squeezed_sol)
        cost, p_min, _, _ = self.simulator.get_stats(final_indices)
        p_surplus = p_min - self.simulator.config.h_min
        if p_surplus < 0: return float('inf'), -float('inf'), None
        score = cost - (p_surplus * self.adaptive_bonus)
        return score, cost, squeezed_sol

    def solve_standalone(self):
        self._tabu_fingerprints.clear()
        print("[AnalyticalSolver] 1. Generating Diverse Seeds...")
        seeds = self._make_diverse_seeds()
        active_pool = []
        for s in seeds:
            idx = self._get_current_indices(s)
            c, p, _, _ = self.simulator.get_stats(idx)
            p_surplus = p - self.simulator.config.h_min
            score = c - (p_surplus * self.adaptive_bonus)
            active_pool.append((score, c, s))
            self._tabu_fingerprints.add(self._get_fingerprint(s, c))
            
        global_best_cost = min(x[1] for x in active_pool)
        global_best_sol = next(x[2] for x in active_pool if x[1] == global_best_cost)
        print(f"   > Seeds initialized. Best Start: {global_best_cost/1e6:.4f}M$")

        BEAM_WIDTH = 5
        MAX_ROUNDS = 50 # Increased rounds for deep optimization
        SINGLE_CANDIDATES = 6
        PAIR_CANDIDATES = 3
        
        stagnation_counter = 0
        kick_tabu_set = set()

        for round_idx in range(MAX_ROUNDS):
            
            if round_idx > 0 and round_idx % 6 == 0:
                kick_tabu_set.clear()
                self._log("   [INFO] Auto-resetting Kick Tabu list.")
            
            next_gen = []
            
            # SWAP
            swapped = self._swap_search(global_best_sol)
            idx = self._get_current_indices(swapped)
            c, p, _, _ = self.simulator.get_stats(idx)
            if c < global_best_cost:
                 diff = global_best_cost - c
                 global_best_cost = c
                 global_best_sol = swapped
                 print(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({global_best_cost/1e6:.4f}M$)")
                 stagnation_counter = 0
                 kick_tabu_set.clear()
                 p_surplus = p - self.simulator.config.h_min
                 score = c - (p_surplus * self.adaptive_bonus)
                 active_pool.insert(0, (score - 500000, c, swapped))

            # FORCE STRATEGY CYCLE
            if stagnation_counter >= 1 or round_idx % 2 == 0:
                # Cycle: TOPO -> BOTTLENECK -> TOPO -> SHOCK
                cycle = round_idx % 4
                if cycle == 0: strategy = "TOPO-DIV"
                elif cycle == 1: strategy = "BOTTLENECK"
                elif cycle == 2: strategy = "LOOP_BALANCE" # <-- НОВА СТРАТЕГІЯ
                elif cycle == 3: strategy = "TOPO-DIV"
                else: strategy = "SHOCK"
                
                self._log(f"[FORCE] Applying '{strategy}'...")
                
                forced_sol = None
                locked = None
                path_sig = None
                
                if strategy == "SHOCK":
                    forced_sol, locked, log_msg = self._forcing_hand_kick(global_best_sol)
                elif strategy == "BOTTLENECK":
                    forced_sol, locked, log_msg = self._upstream_bottleneck_kick(global_best_sol)
                elif strategy == "LOOP_BALANCE":
                    forced_sol, locked, log_msg = self._loop_balancing_kick(global_best_sol)
                else:
                    forced_sol, locked, log_msg, path_sig = self._topological_inversion_kick(global_best_sol, kick_tabu_set)
                
                if forced_sol and locked:
                    self._log(f"     -> {log_msg}")
                    
                    self._log("     [Phase 1: STARVE - Aggressive Reduction]")
                    # Squeeze має знати про компенсацію
                    starved_sol = self._gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=8, verbose=True)
                    
                    self._log("     [Phase 2: HEAL - Final Polish]")
                    final_sol = self._gradient_squeeze(starved_sol, verbose=True)
                    
                    idx = self._get_current_indices(final_sol)
                    c, p, _, _ = self.simulator.get_stats(idx)
                    
                    if p >= self.simulator.config.h_min:
                        p_surplus = p - self.simulator.config.h_min
                        score = c - (p_surplus * self.adaptive_bonus)
                        active_pool.insert(0, (score - 2000000, c, final_sol))
                        self._log(f"     -> Injection Successful: {c/1e6:.4f}M$")
                        stagnation_counter = 0
                        if path_sig: kick_tabu_set.add(path_sig)
                    else:
                        self._log(f"     -> Injection Failed (Infeasible)")
                        stagnation_counter += 1
                        if path_sig: kick_tabu_set.add(path_sig)
                else:
                    self._log(f"     -> No valid kick generated.")
                    stagnation_counter += 1

            # BEAM SEARCH
            total_tasks = len(active_pool) * (SINGLE_CANDIDATES + PAIR_CANDIDATES)
            processed = 0
            sys.stdout.write(f"   > [R{round_idx+1}] Searching... 0%")
            sys.stdout.flush()

            for parent_score, parent_cost, parent_sol in active_pool:
                curr_indices = self._get_current_indices(parent_sol)
                unit_losses = self.simulator.get_heuristics(curr_indices)
                high_friction = sorted(range(self.num_pipes), key=lambda i: unit_losses[i], reverse=True)
                low_friction = sorted(range(self.num_pipes), key=lambda i: unit_losses[i]) 
                
                for pipe_idx in high_friction[:SINGLE_CANDIDATES]:
                    s, c, sol = self._evaluate_candidate(parent_sol, [pipe_idx], "upgrade")
                    if sol: next_gen.append((s, c, sol))
                    processed += 1
                
                for pipe_idx in low_friction[:4]:
                     s, c, sol = self._evaluate_candidate(parent_sol, [pipe_idx], "downgrade")
                     if sol: next_gen.append((s, c, sol))
                     processed += 1

                if len(high_friction) >= 2:
                    top_4 = high_friction[:4]
                    for p1, p2 in itertools.combinations(top_4, 2):
                        if processed >= total_tasks: break 
                        s, c, sol = self._evaluate_candidate(parent_sol, [p1, p2], "upgrade")
                        if sol: next_gen.append((s, c, sol))
                        processed += 1
                
                if processed % 10 == 0 and total_tasks > 0:
                    percent = min(100, int(processed/total_tasks*100))
                    sys.stdout.write(f"\r   > [R{round_idx+1}] Searching... {percent}%")
                    sys.stdout.flush()

            sys.stdout.write(f"\r   > [R{round_idx+1}] Searching... 100%\n")
            
            if not next_gen:
                stagnation_counter = 10
                continue

            next_gen.sort(key=lambda x: x[0]) 
            unique_next_pool = []
            found_new_record = False
            
            for rank, (score, cost, sol) in enumerate(next_gen):
                fp = self._get_fingerprint(sol, cost)
                if fp in self._tabu_fingerprints: continue
                
                if len(unique_next_pool) < BEAM_WIDTH:
                    if rank == 0: refined_sol = self._gradient_squeeze(sol)
                    elif rank < 3: refined_sol = self._gradient_squeeze(sol, max_passes=4, quick_mode=True)
                    else: refined_sol = sol 

                    idx_sol = self._get_current_indices(refined_sol)
                    real_cost, p_min, _, _ = self.simulator.get_stats(idx_sol)
                    
                    if p_min >= self.simulator.config.h_min:
                        real_p_surplus = p_min - self.simulator.config.h_min
                        real_score = real_cost - (real_p_surplus * self.adaptive_bonus)
                        unique_next_pool.append((real_score, real_cost, refined_sol))
                        self._tabu_fingerprints.add(self._get_fingerprint(refined_sol, real_cost))
                        
                        if real_cost < global_best_cost:
                            diff = global_best_cost - real_cost
                            global_best_cost = real_cost
                            global_best_sol = refined_sol
                            found_new_record = True
                            print(f"   > [R{round_idx+1}] 💎 New Record: -${diff:,.0f} ({global_best_cost/1e6:.4f}M$)")

            if found_new_record:
                stagnation_counter = 0
                kick_tabu_set.clear()
            else:
                stagnation_counter += 1
                if stagnation_counter % 8 == 0:
                    kick_tabu_set.clear()
                    self._log("   [INFO] Tabu list cleared due to stagnation.")

            if not unique_next_pool:
                stagnation_counter += 2 
            else:
                unique_next_pool.sort(key=lambda x: x[0])
                active_pool = unique_next_pool[:BEAM_WIDTH]

        print(f"[AnalyticalSolver] Фінал: {global_best_cost/1e6:.4f}M$")
        self.best_found_solution = global_best_sol 
        return global_best_sol

