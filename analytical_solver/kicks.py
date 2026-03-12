import random
import networkx as nx
import numpy as np
import itertools

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
    
    def vns_structured_kick(self, indices, stagnation_level):
        """Variable Neighborhood Search: збільшує розмір і радіус удару зі зростанням стагнації"""
        n = self.ctx.num_pipes
        # Розміри околиці залежно від рівня стагнації (від 5% до 25%)
        neighborhood_sizes = [0.05, 0.10, 0.15, 0.20, 0.25]
        k_idx = min(stagnation_level // 2, len(neighborhood_sizes) - 1)
        pct = neighborhood_sizes[k_idx]
        
        n_change = max(3, int(n * pct))
        
        # Структурована зміна: вибираємо найгірші труби за unit_loss
        unit_losses = self.ctx.get_cached_heuristics(indices)
        worst_pipes = sorted(range(n), key=lambda i: unit_losses[i], reverse=True)[:n_change + max(5, n//10)]
        target_pipes = random.sample(worst_pipes, n_change)
        
        kicked = list(indices)
        locked = set()
        
        # Сила удару теж зростає зі стагнацією
        jump_size = 1 if k_idx < 2 else (2 if k_idx < 4 else 3)
        
        for p in target_pipes:
            direction = random.choice([-jump_size, jump_size])
            kicked[p] = max(0, min(self.ctx.max_d_idx, kicked[p] + direction))
            locked.add(p)
            
        healed, ok, boosts = self.ls.heal_network(kicked, locked)
        if not ok: return None, None, ""
        
        return healed, locked, f"VNS-KICK (Level {k_idx}): Shifted {n_change} high-loss pipes by ±{jump_size}. Healed {boosts}x."

    # 🔴 ВИПРАВЛЕНО: Додано аргумент dyn_bonus
    def segment_restart_kick(self, indices, dyn_bonus):
        """Segment Restart: Заморожує частину і скидає решту до максимуму"""
        n = self.ctx.num_pipes
        
        freeze_pct = 0.50 if n <= 50 else 0.20
        n_freeze = max(5, int(n * freeze_pct)) 
        
        # 🔴 ПАТЧ 1: Додаємо рандомізацію у вибір заморожених труб
        top_candidates = self.ls.get_high_impact_pipes(indices, n_freeze + max(3, n // 10))
        frozen_pipes = set(random.sample(top_candidates, min(n_freeze, len(top_candidates))))
        
        fresh = list(indices)
        for p in range(n):
            if p not in frozen_pipes:
                # 🔴 ПАТЧ 1: Рандомізація скидання (10% шанс не йти до самого максимуму)
                if random.random() < 0.1 and self.ctx.max_d_idx > 0:
                    fresh[p] = self.ctx.max_d_idx - 1
                else:
                    fresh[p] = self.ctx.max_d_idx 
                
        squeezed = self.ls.gradient_squeeze(fresh, locked_pipes=frozen_pipes, max_passes=None, quick_mode=False, dyn_bonus=dyn_bonus)
        
        cost, p_val, feas, _ = self.ctx.get_cached_stats(squeezed)
        if not feas or p_val < self.ctx.simulator.config.h_min:
            return None, None, ""
            
        return squeezed, frozen_pipes, f"SEGMENT-RESTART: Froze {len(frozen_pipes)} pipes, Top-Down squeeze."

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
        if not ok: return None, None, ""
        
        return healed, locked, f"IPC-CROSSOVER: Merged with peer (Mine: {p_mine:.0%}, Peer: {1-p_mine:.0%})."
    
    def corridor_search_kick(self, sol_A, sol_B):
        """Досліджує простір між двома рішеннями (Hamming 30-70)"""
        child = list(sol_A)
        diff_pipes = [i for i in range(self.ctx.num_pipes) if sol_A[i] != sol_B[i]]
            
        locked = set()
        for i in diff_pipes:
            # 🔴 Зберігаємо фізичну структуру: беремо ген від А або Б, а не середнє
            if random.random() < 0.5:
                child[i] = sol_B[i]
            locked.add(i)
            
        healed, ok, boosts = self.ls.heal_network(child, locked)
        if not ok: return None, None, ""
        
        return healed, locked, f"CORRIDOR: Interpolated {len(diff_pipes)} diff pipes. Healed {boosts}x."