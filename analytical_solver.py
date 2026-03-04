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

    def _gradient_squeeze(self, solution, locked_pipes=None, max_passes=None, quick_mode=False):
        if locked_pipes is None: locked_pipes = set()
        current_solution = list(solution)
        improved = True
        
        lengths = self.simulator.lengths
        costs_array = self.simulator.costs 
        passes = 0
        
        active_indices = [i for i in range(self.num_pipes) if i not in locked_pipes]
        
        while improved:
            improved = False
            passes += 1
            if max_passes and passes > max_passes: break
            
            curr_indices = self._get_current_indices(current_solution)
            unit_losses = self.simulator.get_heuristics(curr_indices)
            avg_loss = sum(unit_losses) / len(unit_losses) if len(unit_losses) > 0 else 0
            
            savings_candidates = []
            for i in active_indices:
                if quick_mode and unit_losses[i] > avg_loss * 1.5: continue
                
                current_idx = curr_indices[i]
                if current_idx > 0:
                    dollar_save = lengths[i] * (costs_array[current_idx] - costs_array[current_idx - 1])
                    if dollar_save < 50.0: continue 
                    
                    risk = unit_losses[i] + 1e-9
                    score = dollar_save / risk
                    savings_candidates.append((i, score))
            
            if not savings_candidates: break

            savings_candidates.sort(key=lambda x: x[1], reverse=True)
            limit = 5 if quick_mode else 20
            candidates_to_try = savings_candidates[:limit]

            batch_count = 0
            max_batch = 5 
            
            if quick_mode and len(candidates_to_try) >= 3:
                test_batch = list(current_solution)
                batch_indices = []
                
                for idx, _ in candidates_to_try[:3]:
                    c_idx = self.diameters.index(test_batch[idx])
                    test_batch[idx] = self.diameters[c_idx - 1]
                    batch_indices.append(idx)
                
                _, is_feasible = self.simulator.get_hydraulic_state(test_batch)
                
                if is_feasible:
                    current_solution = test_batch
                    improved = True
                    continue 
            
            for idx, _ in candidates_to_try:
                current_idx = self.diameters.index(current_solution[idx])
                test_solution = list(current_solution)
                test_solution[idx] = self.diameters[current_idx - 1]
                
                _, is_feasible = self.simulator.get_hydraulic_state(test_solution)
                
                if is_feasible:
                    current_solution = test_solution
                    improved = True
                    batch_count += 1
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

        source = self.simulator.sources[0]
        try:
            path_nodes = nx.shortest_path(self.simulator.graph, source, crit_node)
        except: return solution, set(), ""
            
        kicked = list(solution)
        locked = set()
        
        path_pipes = []
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_data = self.simulator.graph.get_edge_data(u, v)
            for key, attr in edge_data.items():
                if key in self.simulator.component_names:
                    idx = self.simulator.component_names.index(key)
                    path_pipes.append(idx)
                    break
        
        for idx in path_pipes:
            curr_idx = self.diameters.index(kicked[idx])
            if curr_idx < len(self.diameters) - 1:
                kicked[idx] = self.diameters[curr_idx + 1]
                locked.add(idx) 
        
        log_msg = f"SHOCK: Upgraded Critical Path {self._indices_to_path_str(path_pipes)}"
        return kicked, locked, log_msg

    def _topological_inversion_kick(self, solution, tabu_set=None):
        """
        TOPOLOGICAL DIVERSITY SEARCH (with Tabu):
        """
        if tabu_set is None: tabu_set = set()
        
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
            candidates = list(itertools.islice(path_generator, 50))
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
            
            # Signature for Tabu: Sorted tuple of unique indices
            signature = tuple(sorted(p_indices))
            if signature in tabu_set: continue # SKIP TABU PATHS
            
            scored_paths.append({'indices': p_indices, 'overlap': overlap_ratio, 'signature': signature})

        scored_paths.sort(key=lambda x: x['overlap'])
        
        if not scored_paths: return None, None, "No non-tabu alternatives", None

        # Take best available
        best_alt = scored_paths[0]
        
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
        
        for h_pipe in high_f[:12]:
            for l_pipe in low_f[:12]:
                if h_pipe == l_pipe: continue
                d_high = best_sol[h_pipe]
                d_low  = best_sol[l_pipe]
                h_idx = self.diameters.index(d_high)
                l_idx = self.diameters.index(d_low)
                
                if h_idx < len(self.diameters)-1 and l_idx > 0:
                    test_sol = list(best_sol)
                    test_sol[h_pipe] = self.diameters[h_idx + 1] 
                    test_sol[l_pipe] = self.diameters[l_idx - 1]
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
        MAX_ROUNDS = 30 
        SINGLE_CANDIDATES = 6
        PAIR_CANDIDATES = 3
        
        stagnation_counter = 0
        kick_tabu_set = set() # NEW: Tabu for failed kicks

        for round_idx in range(MAX_ROUNDS):
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
                 kick_tabu_set.clear() # Clear tabu on success
                 p_surplus = p - self.simulator.config.h_min
                 score = c - (p_surplus * self.adaptive_bonus)
                 active_pool.insert(0, (score - 500000, c, swapped))

            # FORCE
            if stagnation_counter >= 1 or round_idx % 2 == 0:
                strategy = "SHOCK" if round_idx % 4 == 0 else "TOPO-DIV"
                self._log(f"[FORCE] Applying '{strategy}'...")
                
                forced_sol = None
                locked = None
                path_sig = None
                
                if strategy == "SHOCK":
                    forced_sol, locked, log_msg = self._forcing_hand_kick(global_best_sol)
                else:
                    forced_sol, locked, log_msg, path_sig = self._topological_inversion_kick(global_best_sol, kick_tabu_set)
                
                if forced_sol and locked:
                    self._log(f"     -> {log_msg}")
                    starved_sol = self._gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=8)
                    final_sol = self._gradient_squeeze(starved_sol)
                    idx = self._get_current_indices(final_sol)
                    c, p, _, _ = self.simulator.get_stats(idx)
                    
                    if p >= self.simulator.config.h_min:
                        p_surplus = p - self.simulator.config.h_min
                        score = c - (p_surplus * self.adaptive_bonus)
                        active_pool.insert(0, (score - 2000000, c, final_sol))
                        self._log(f"     -> Injection Successful: {c/1e6:.4f}M$")
                        stagnation_counter = 0
                        # If TOPO-DIV was successful in injection but costly, we still might want to try other paths later
                        # But for now, let's tabu this one so we don't spam it if it doesn't yield a record quickly
                        if path_sig: kick_tabu_set.add(path_sig)
                    else:
                        self._log(f"     -> Injection Failed (Infeasible)")
                        stagnation_counter += 1
                        if path_sig: kick_tabu_set.add(path_sig) # Ban infeasible paths too
                else:
                    self._log(f"     -> No valid kick generated (All tabu or none found).")
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
                kick_tabu_set.clear() # Clear tabu on record
            else:
                stagnation_counter += 1

            if not unique_next_pool:
                stagnation_counter += 2 
            else:
                unique_next_pool.sort(key=lambda x: x[0])
                active_pool = unique_next_pool[:BEAM_WIDTH]

        print(f"[AnalyticalSolver] Фінал: {global_best_cost/1e6:.4f}M$")
        self.best_found_solution = global_best_sol 
        return global_best_sol