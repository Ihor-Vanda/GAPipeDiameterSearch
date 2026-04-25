import random
import networkx as nx
import numpy as np
import itertools

class KickStrategies:
    def __init__(self, ctx, local_search):
        self.ctx = ctx
        self.ls = local_search

    def forcing_hand_kick(self, indices, T, **kwargs):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return indices, set(), ""

        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return indices, set(), ""
            
        kicked, locked = list(indices), set()
        
        aggressiveness = 0.05 + (0.35 * T)
        limit = max(1, int(len(path_pipes) * aggressiveness))
        
        unit_losses = self.ctx.get_cached_heuristics(indices)
        worst_pipes = sorted(path_pipes, key=lambda i: unit_losses[i], reverse=True)[:limit]
        
        for idx in worst_pipes:
            if kicked[idx] < self.ctx.max_d_idx:
                kicked[idx] += 1
                locked.add(idx) 
        
        if not locked: return indices, set(), ""
        return kicked, locked, f"SHOCK: Upgraded {len(locked)} worst pipes on Critical Path (T={T:.2f})"

    def upstream_bottleneck_kick(self, indices, T, **kwargs):
        failed_pipes = kwargs.get('failed_pipes', {})
        boosted_pipes = kwargs.get('boosted_pipes', {})
        current_round = kwargs.get('current_round', 0)
        tenure = 5
        
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return indices, set(), "", -1
        
        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return indices, set(), "", -1
        
        kicked, locked = list(indices), set()
        
        valid_taper_violations = []
        for i in range(len(path_pipes) - 1, 0, -1):
            curr_p, prev_p = path_pipes[i], path_pipes[i-1]
            
            if curr_p in failed_pipes and (current_round - failed_pipes[curr_p]) < 10: 
                continue
                
            if indices[curr_p] < indices[prev_p]:
                if kicked[curr_p] < self.ctx.max_d_idx:
                    valid_taper_violations.append(curr_p)

        if valid_taper_violations:
            target_pipe = None
            for p in valid_taper_violations:
                if p not in boosted_pipes or (current_round - boosted_pipes[p]) >= tenure:
                    target_pipe = p
                    break
            
            if target_pipe is None:
                target_pipe = valid_taper_violations[0]

            kicked[target_pipe] += 1
            locked.add(target_pipe)
            return kicked, locked, f"BOTTLENECK: Boosted Pipe {target_pipe} (Taper Violation)", target_pipe
                    
        unit_losses = self.ctx.get_cached_heuristics(indices)
        search_depth = max(1, int(len(path_pipes) * (0.3 + 0.4 * T)))
        
        base_candidates = [(idx, unit_losses[idx]) for idx in path_pipes[:search_depth] 
                           if indices[idx] < self.ctx.max_d_idx 
                           and not (idx in failed_pipes and (current_round - failed_pipes[idx]) < 10)]
        
        if not base_candidates: return indices, set(), "", -1

        candidates = [c for c in base_candidates 
                      if c[0] not in boosted_pipes or (current_round - boosted_pipes[c[0]]) >= tenure]
        
        if not candidates:
            candidates = base_candidates
        
        boost_pct = 0.05 + (0.15 * T)
        n_boost = max(1, int(len(candidates) * boost_pct))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_pipes = [x[0] for x in candidates[:n_boost]]
        
        for p in best_pipes:
            kicked[p] += 1
            locked.add(p)
            
        return kicked, locked, f"BOTTLENECK: Boosted {len(best_pipes)} Max-Loss Pipes.", best_pipes[0]

    def topological_inversion_kick(self, indices, T, **kwargs):
        tabu_set = kwargs.get('tabu_set', set())
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
        for _ in range(5):
            try:
                path = nx.shortest_path(temp_G, source, crit_node, weight='weight')
                if path != dom_nodes: candidates.append(path)
                for u, v in zip(path[:-1], path[1:]): temp_G[u][v]['weight'] *= 5.0
            except nx.NetworkXNoPath: break

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
            signature = tuple(sorted(p_indices))
            if signature in tabu_set: continue
            scored_paths.append({'indices': p_indices, 'overlap': overlap / length, 'signature': signature})

        scored_paths.sort(key=lambda x: x['overlap'])
        if not scored_paths: return None, None, "All paths tabu", None

        best_alt = scored_paths[0]
        kicked, locked = list(indices), set()
        
        aggressiveness = 0.10 + (0.50 * T)
        max_pipes = max(2, int(len(best_alt['indices']) * aggressiveness))
        
        indices_to_change = best_alt['indices']
        if len(indices_to_change) > max_pipes:
             indices_to_change = random.sample(indices_to_change, max_pipes)

        for idx in indices_to_change:
            force_idx = min(target_capacity_idx, self.ctx.max_d_idx)
            
            if kicked[idx] < force_idx:
                kicked[idx] = force_idx
                locked.add(idx)
            elif kicked[idx] == force_idx: 
                locked.add(idx)

        return kicked, locked, f"TOPO-INV: Boosted Alt Path (Overlap {best_alt['overlap']:.2f}, T={T:.2f})", best_alt['signature']

    def loop_balancing_kick(self, indices, T, **kwargs):
        failed_pipes = kwargs.get('failed_pipes', {})
        current_round = kwargs.get('current_round', 0)
        dyn_bonus = kwargs.get('dyn_bonus', 0)
        
        LOOP_BALANCE_PIPE_TENURE = min(80, max(3, self.ctx.num_pipes // 5))
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, "", -1

        if not hasattr(self, '_cached_cycles'):
            try: self._cached_cycles = nx.cycle_basis(self.ctx.base_G_flow)
            except: self._cached_cycles = []
        cycles = self._cached_cycles
        
        if not cycles: return None, None, "No cycles found", -1

        max_allowed_drop = 1 + int(2 * T) 

        best_drop_achieved = -1
        candidates = []
        random.shuffle(cycles)

        for cycle_nodes in cycles:
            cycle_indices = []
            full_cycle = cycle_nodes + [cycle_nodes[0]]
            for u, v in zip(full_cycle[:-1], full_cycle[1:]):
                if (u, v) in self.ctx.edge_to_pipe: 
                    cycle_indices.append(self.ctx.edge_to_pipe[(u, v)])

            if not cycle_indices: continue

            restrict_pct = 0.10 + (0.30 * T)
            n_restrict = max(1, int(len(cycle_indices) * restrict_pct))

            for _ in range(3):
                chosen_pipes = random.sample(cycle_indices, n_restrict)

                if any((current_round - failed_pipes.get(p, -999)) < LOOP_BALANCE_PIPE_TENURE for p in chosen_pipes): 
                    continue
                if any(indices[p] < 2 for p in chosen_pipes): 
                    continue 
                
                max_drop = min(max_allowed_drop, min(indices[p] for p in chosen_pipes)) 
                
                for drop in range(max_drop, 0, -1):
                    if drop < best_drop_achieved and n_restrict == 1: continue 
                    
                    kicked, locked = list(indices), set()
                    
                    for p in chosen_pipes:
                        kicked[p] -= drop
                        locked.add(p)
                    
                    healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
                    if is_feasible:
                        test_squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
                        sq_cost, _, _, _ = self.ctx.get_cached_stats(test_squeezed)
                        
                        candidates.append((sq_cost, healed_sol, locked, 
                                           f"FLOW STEER: Cut {n_restrict} pipes in loop (-{drop}). Healed {boosts}x.", 
                                           chosen_pipes[0]))
                        
                        best_drop_achieved = max(best_drop_achieved, drop)
                        break
            
            if len(candidates) >= 5: break

        if not candidates: return None, None, "FLOW STEER: Exhaustive search found no valid bypass.", -1
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])
        return chosen[1], chosen[2], chosen[3], chosen[4]

    def zero_sum_shift_kick(self, indices, T, **kwargs):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        zs_tabu = kwargs.get('zero_sum_tabu', {})
        curr_round = kwargs.get('current_round', 0)

        unit_losses = self.ctx.get_cached_heuristics(indices)
        kicked = list(indices)
        
        upgrades = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] < self.ctx.max_d_idx:
                c_up = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]+1] - self.ctx.costs_array[kicked[i]])
                upgrades.append((i, c_up, unit_losses[i] / max(c_up, 1.0)))
        upgrades.sort(key=lambda x: x[2], reverse=True)

        valid_upgrades = []
        for up in upgrades:
            if curr_round - zs_tabu.get(up[0], -999) > 15:
                valid_upgrades.append(up)

        downgrades = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] > 0:
                c_down = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]] - self.ctx.costs_array[kicked[i]-1])
                downgrades.append((i, c_down, c_down / max(unit_losses[i], 1e-5)))
        downgrades.sort(key=lambda x: x[2], reverse=True)

        best_sol, best_locked, best_cost = None, None, float('inf')
        
        search_pool_size = max(15, self.ctx.num_pipes // 10)
        
        tests_limit = 5 if T < 0.3 else 2
        
        base_down = max(1, self.ctx.num_pipes // 50)
        max_downgrades = base_down if T < 0.2 else (base_down * 3 if T < 0.5 else base_down * 6)
        
        valid_found = 0

        for up_idx, cost_invest, _ in valid_upgrades[:search_pool_size]:
            if valid_found >= tests_limit: break
            
            test_sol = list(indices)
            test_sol[up_idx] += 1
            locked_set = {up_idx}
            savings = 0
            down_count = 0
            
            for down_idx, cost_save, _ in downgrades:
                if down_idx == up_idx or down_idx in locked_set: continue
                
                test_sol[down_idx] -= 1 
                locked_set.add(down_idx)
                savings += cost_save
                down_count += 1
                
                if savings > cost_invest:
                    healed_sol, ok, _ = self.ls.heal_network(test_sol, locked_set)
                    if ok:
                        c, _, _, _ = self.ctx.get_cached_stats(healed_sol)
                        if c < best_cost:
                            best_cost, best_sol, best_locked, best_up = c, healed_sol, locked_set, up_idx
                        valid_found += 1
                    break 
                    
                if down_count >= max_downgrades:
                    break 

        if best_sol:
            mode_str = "1-to-1" if len(best_locked) == 2 else f"1-to-{len(best_locked)-1}"
            return best_sol, best_locked, f"ZERO-SUM ({mode_str}): P{best_up} upgraded. Cost: {best_cost/1e6:.4f}M$"
            
        return None, None, ""
    
    def peripheral_trim_kick(self, indices, T, **kwargs):
        indices_copy, periphery_pipes = self.ctx.get_lazy_periphery(indices)
        if not periphery_pipes: 
            return None, None, "No targets"
        
        trim_pct = 0.02 + (0.13 * T)
        pipes_to_cut = max(1, int(len(periphery_pipes) * trim_pct))
        
        candidates = []
        combo_limit = min(len(periphery_pipes), max(10, self.ctx.num_pipes // 2)) 
        max_checks = min(25, max(5, self.ctx.num_pipes // 10))
        dyn_bonus = kwargs.get('dyn_bonus', 0)
        
        for _ in range(max_checks):
            combo = random.sample(periphery_pipes[:combo_limit], pipes_to_cut)
            
            valid_combo = False
            for p in combo:
                if indices_copy[p] > 0:
                    valid_combo = True
                    break
            
            if not valid_combo: 
                continue
            
            test_sol = list(indices_copy)
            locked = set()
            
            for p in combo: 
                if test_sol[p] > 0:
                    test_sol[p] -= 1
                    locked.add(p)
            
            healed_sol, is_feasible, boosts = self.ls.heal_network(test_sol, locked)
            if is_feasible:
                h_c, h_p, _, _ = self.ctx.get_cached_stats(healed_sol) 
                h_score = h_c - ((h_p - self.ctx.simulator.config.h_min) * dyn_bonus)
                candidates.append((h_score, healed_sol, locked, boosts))
                
        if not candidates: 
            return None, None, "No valid trim candidates found or heal failed"
        
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])
        
        return chosen[1], chosen[2], f"TRIM: Shrunk {len(chosen[2])} peripheral pipes. Healed {chosen[3]}x."

    def smart_perturbation_kick(self, indices, T, **kwargs):        
        base_mu = kwargs.get('mu_perturb_pct', 0.10)
        dynamic_mu = base_mu + (0.15 * T)
        dynamic_sigma = 0.02 + (0.05 * T)
        target_pct = random.gauss(dynamic_mu, dynamic_sigma)
        target_pct = max(0.01, min(0.30, target_pct))
        
        max_perturb = kwargs.get('max_perturb', self.ctx.num_pipes)
        n_perturb = min(max_perturb, max(1, int(self.ctx.num_pipes * target_pct)))
        
        kicked, locked = list(indices), set()
        candidates = [i for i in range(self.ctx.num_pipes) if 0 < kicked[i] < self.ctx.max_d_idx]
        if len(candidates) < n_perturb: candidates = list(range(self.ctx.num_pipes))
            
        chosen = random.sample(candidates, min(n_perturb, len(candidates)))
        
        max_delta = 2 if T > 0.6 else 1
        
        for p_idx in chosen:
            if max_delta > 1: delta = random.choices([-2, -1, 1, 2], weights=[1, 3, 3, 1])[0]
            else: delta = random.choice([-1, 1])
                
            new_val = max(0, min(self.ctx.max_d_idx, kicked[p_idx] + delta))
            kicked[p_idx] = new_val
            if delta > 0: locked.add(p_idx) 
                
        healed, ok, boosts = self.ls.heal_network(kicked, locked)
        if not ok: return None, None, ""
        return healed, locked, f"SMART-PERTURB: Shifted {len(chosen)} pipes (Size: {target_pct:.1%}). Healed {boosts}x.", target_pct

    def ruin_and_recreate_kick(self, indices, T, **kwargs):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        ruin_center = crit_node if T < 0.8 else random.choice(list(self.ctx.base_G_flow.nodes()))
        
        base_mu = kwargs.get('mu_ruin_pct', 0.05)
        dynamic_sigma = 0.01 + (0.04 * T)
        target_pct = random.gauss(base_mu, dynamic_sigma)
        target_pct = max(0.01, min(0.25, target_pct))
        
        target_pipes = max(3, int(self.ctx.num_pipes * target_pct))
        
        n = self.ctx.num_pipes
        cutoff = 3 if n < 50 else (4 if n < 200 else (6 if n < 1000 else 8))
        
        pipe_with_dist, seen_pipes = [], set()
        
        try:
            for node, dist in nx.single_source_shortest_path_length(self.ctx.base_G_flow, ruin_center, cutoff=cutoff).items():
                for p in self.ctx.node_to_pipes.get(node, []):
                    if p not in seen_pipes:
                        seen_pipes.add(p)
                        pipe_with_dist.append((dist, p))
        except: pass

        if not pipe_with_dist: return None, None, ""
        pipe_with_dist.sort(key=lambda x: x[0])
        cluster_pipes = set(p for _, p in pipe_with_dist[:target_pipes])

        kicked, ruined_count = list(indices), 0
        max_catalog_drop = max(1, self.ctx.max_d_idx // 3) 
        temp_drop = 1 + int(2 * T)
        max_drop = min(temp_drop, max_catalog_drop)
        
        for p in cluster_pipes:
            if kicked[p] > 0:
                kicked[p] = max(0, kicked[p] - random.randint(1, max_drop))
                ruined_count += 1

        if ruined_count == 0: return None, None, ""

        healed_sol, is_feas, boosts = self.ls.heal_network(kicked, set())
        if not is_feas: return None, None, ""
        
        return healed_sol, set(), f"R&R: Downgraded {ruined_count} pipes (Size: {target_pct:.1%}). Healed {boosts}x.", target_pct
    
    def basin_escape(self, indices, T, **kwargs):
        global_archive = kwargs.get('global_archive', [])
        if not global_archive or len(global_archive) < 2:
            return None, None, "Archive too small"

        num_pipes = len(indices)
        min_dist_threshold = max(5, int(num_pipes * 0.05))
        
        target_sol = None
        max_dist = -1
        
        for arch_cost, arch_sol in global_archive:
            dist = sum(1 for a, b in zip(indices, arch_sol) if a != b)
            
            if dist > max_dist and dist >= min_dist_threshold:
                max_dist = dist
                target_sol = arch_sol
                
        if target_sol is None:
            return None, None, f"Archive degenerated. No distinct basins found."

        diff_pipes = [i for i in range(num_pipes) if indices[i] != target_sol[i]]
        
        base_mu = kwargs.get('mu_escape_pct', 0.20)
        target_pct = max(0.05, min(0.60, random.gauss(base_mu + (0.20 * T), 0.05 + (0.05 * T))))
        
        n_replace = max(1, int(len(diff_pipes) * target_pct))
        replace_pipes = random.sample(diff_pipes, n_replace)
        kicked = list(indices)
        for p in replace_pipes: kicked[p] = target_sol[p]

        cost_limit = kwargs.get('cost_limit', float('inf'))
        healed, ok, boosts = self.ls.heal_network(kicked, set())
        if not ok: 
            return None, None, "Heal failed after transplant"
            
        log_msg = f"BASIN-ESCAPE: Transplanted {n_replace} pipes (Size: {target_pct:.1%}). Healed {boosts}x."
        return healed, set(), log_msg, target_pct
    
    def spatial_perturb_kick(self, indices, **kwargs):
        T = kwargs.get('T', 0.5)
        num_pipes = len(indices)
        
        base_mu = kwargs.get('mu_spatial_pct', 0.20)
        dynamic_sigma = 0.02 + (0.10 * T) 
        target_pct = max(0.02, min(0.60, random.gauss(base_mu, dynamic_sigma)))
        
        target_size = max(2, int(num_pipes * target_pct))
        
        max_p = kwargs.get('max_perturb', int(num_pipes * 0.50))
        target_size = min(target_size, max_p)

        import networkx as nx
        physical_G = nx.Graph(self.ctx.base_G_flow) 

        local_pipes = set()
        
        for _ in range(5):
            epicenter_node = random.choice(list(physical_G.nodes()))
            
            radius = 1
            while len(local_pipes) < target_size and radius < 20:
                subgraph = nx.ego_graph(physical_G, epicenter_node, radius=radius, undirected=True)
                
                for u, v in subgraph.edges():
                    key = (u, v)
                    if key in self.ctx.edge_to_pipe:
                        local_pipes.add(self.ctx.edge_to_pipe[key])
                    elif (v, u) in self.ctx.edge_to_pipe:
                        local_pipes.add(self.ctx.edge_to_pipe[(v, u)])
                        
                radius += 1
                
                if radius > 1 and len(subgraph.nodes()) == len(nx.ego_graph(physical_G, epicenter_node, radius=radius-1).nodes()):
                    break
                    
            if len(local_pipes) >= target_size:
                break 

        if len(local_pipes) < target_size:
            local_pipes = set(random.sample(range(num_pipes), target_size))

        local_pipes = list(local_pipes)[:target_size]

        kicked = list(indices)
        for p in local_pipes:
            shift = random.choice([-2, -1, 1, 2])
            kicked[p] = max(1, min(self.ctx.max_d_idx - 1, kicked[p] + shift))

        healed, ok, boosts = self.ls.heal_network(kicked, set())
        if not ok:
            return None, None, "Heal failed after spatial perturb"

        log_msg = f"SPATIAL-PERTURB: Shifted {len(local_pipes)} pipes (Size: {target_pct:.1%}). Healed {boosts}x."
        
        return healed, set(), log_msg, target_pct