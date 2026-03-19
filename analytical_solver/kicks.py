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
        c_current, _, _, _ = self.ctx.get_cached_stats(indices)
        aggressiveness = 0.3 if c_current > 2_200_000 else (0.15 if c_current > 2_050_000 else 0.05)
        limit = max(1, int(len(path_pipes) * aggressiveness))
        unit_losses = self.ctx.get_cached_heuristics(indices)
        
        worst_pipes = sorted(path_pipes, key=lambda i: unit_losses[i], reverse=True)[:limit]
        
        for idx in worst_pipes:
            if kicked[idx] < self.ctx.max_d_idx:
                kicked[idx] += 1
                locked.add(idx) 
        
        if not locked: return indices, set(), ""
        return kicked, locked, f"SHOCK: Upgraded {len(locked)} worst pipes on Critical Path"

    def upstream_bottleneck_kick(self, indices, failed_pipes=None, current_round=0):
        failed_pipes = failed_pipes or {}
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return indices, set(), "", -1
        
        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return indices, set(), "", -1
        
        kicked, locked = list(indices), set()
        bottleneck_found = False
        failed_id = -1
        d_prev, d_curr = -1, -1
        
        # 1. Спроба знайти Taper Violation (звуження)
        for i in range(len(path_pipes) - 1, 0, -1):
            curr_p = path_pipes[i]
            prev_p = path_pipes[i-1]
            
            if curr_p in failed_pipes and (current_round - failed_pipes[curr_p]) < 10:
                continue
                
            d_curr = indices[curr_p]
            d_prev = indices[prev_p]
            
            if d_curr < d_prev:
                if kicked[curr_p] < self.ctx.max_d_idx:
                    kicked[curr_p] += 1
                    locked.add(curr_p)
                    bottleneck_found = True
                    failed_id = curr_p
                    break
                    
        if not bottleneck_found:
            unit_losses = self.ctx.get_cached_heuristics(indices)
            half = max(1, len(path_pipes) // 2)
            candidates = [(idx, unit_losses[idx]) for idx in path_pipes[:half] 
                          if indices[idx] < self.ctx.max_d_idx 
                          and not (idx in failed_pipes and (current_round - failed_pipes[idx]) < 10)]
            
            if not candidates:
                return indices, set(), "", -1
                
            best_pipe = max(candidates, key=lambda x: x[1])[0]
            kicked[best_pipe] = min(self.ctx.max_d_idx, kicked[best_pipe] + 1)
            locked.add(best_pipe)
            failed_id = best_pipe
            return kicked, locked, f"BOTTLENECK: Max-Loss Pipe {best_pipe} boosted.", failed_id

        return kicked, locked, f"BOTTLENECK: Boosted Pipe {failed_id} (Taper Violation (Idx{d_prev}->Idx{d_curr}))", failed_id

    def topological_inversion_kick(self, indices, tabu_set):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, "", None
        
        dom_pipes, dom_nodes = self.ctx.get_dominant_path(indices, crit_node)
        if not dom_nodes: return None, None, "", None
        
        source = dom_nodes[0]
        dom_edges = set(tuple(sorted((u, v))) for u, v in zip(dom_nodes[:-1], dom_nodes[1:]))
        
        target_capacity_idx = int(np.mean([indices[p] for p in dom_pipes])) if dom_pipes else self.ctx.max_d_idx
        
        # 1. Створюємо тимчасовий граф і шукаємо альтернативні шляхи
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

        # 2. Оцінюємо знайдені шляхи
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

        # 3. Вибираємо найкращий і застосовуємо Адаптивну Агресію
        best_alt = scored_paths[0]
        kicked, locked, push_count = list(indices), set(), 0
        
        c_current, _, _, _ = self.ctx.get_cached_stats(indices)
        aggressiveness = 0.4 if c_current > 2_200_000 else (0.2 if c_current > 2_050_000 else 0.08)
        max_pipes_to_change = max(2, int(len(best_alt['indices']) * aggressiveness))
        
        indices_to_change = best_alt['indices']
        if len(indices_to_change) > max_pipes_to_change:
             indices_to_change = random.sample(indices_to_change, max_pipes_to_change)

        for idx in indices_to_change:
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

    def diameter_diversity_kick(self, indices, stagnation_counter):
        kicked, locked = list(indices), set()
        
        base_n = max(5, int(self.ctx.num_pipes * 0.12))
        aggression_step = max(0.5, self.ctx.num_pipes / 100.0) 
        n_to_change = min(self.ctx.num_pipes // 2, base_n + int(stagnation_counter * aggression_step))
        
        pipes_to_change = random.sample(range(self.ctx.num_pipes), n_to_change)
        
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

        if stagnation_counter > 8:
            all_nodes = list(self.ctx.base_G_flow.nodes())
            ruin_center = random.choice(all_nodes)
        else:
            ruin_center = crit_node

        max_pct = 0.20 if self.ctx.num_pipes <= 50 else 0.40
        base_pct = random.uniform(0.10, max_pct / 2)
        target_pct = min(max_pct, base_pct + (stagnation_counter * 0.025)) 
        
        target_pipes = max(3, int(self.ctx.num_pipes * target_pct))
        
        # 🔴 ЛІМІТЕР МАСТАБУ: Не більше 35 труб (для великих мереж)
        target_pipes = min(35, target_pipes)
        
        pipe_with_dist = []
        seen_pipes = set()
        
        try:
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
    
    def vns_structured_kick(self, indices, stagnation_level):
        """Variable Neighborhood Search: збільшує розмір і радіус удару зі зростанням стагнації"""
        n = self.ctx.num_pipes
        neighborhood_sizes = [0.05, 0.10, 0.15, 0.20, 0.25]
        k_idx = min(stagnation_level // 2, len(neighborhood_sizes) - 1)
        pct = neighborhood_sizes[k_idx]
        
        n_change = max(3, int(n * pct))
        hard_limit = 15 if self.ctx.num_pipes >= 200 else 30
        n_change = min(n_change, hard_limit)
        
        unit_losses = self.ctx.get_cached_heuristics(indices)
        worst_pipes = sorted(range(n), key=lambda i: unit_losses[i], reverse=True)[:n_change + max(5, n//10)]
        target_pipes = random.sample(worst_pipes, n_change)
        
        kicked = list(indices)
        locked = set()
        
        jump_size = 1 if k_idx < 2 else (2 if k_idx < 4 else 3)
        
        for p in target_pipes:
            direction = random.choice([-jump_size, jump_size])
            kicked[p] = max(0, min(self.ctx.max_d_idx, kicked[p] + direction))
            locked.add(p)
            
        healed, ok, boosts = self.ls.heal_network(kicked, locked)
        if not ok: return None, None, ""
        
        return healed, locked, f"VNS-KICK (Level {k_idx}): Shifted {n_change} high-loss pipes by ±{jump_size}. Healed {boosts}x."

    def segment_restart_kick(self, indices, dyn_bonus):
        """Segment Restart: Заморожує частину і частково скидає решту"""
        n = self.ctx.num_pipes
        
        freeze_pct = 0.50 if n <= 50 else 0.20
        n_freeze = max(5, int(n * freeze_pct)) 
        
        top_candidates = self.ls.get_high_impact_pipes(indices, n_freeze + max(3, n // 10))
        frozen_pipes = set(random.sample(top_candidates, min(n_freeze, len(top_candidates))))
        
        fresh = list(indices)
        for p in range(n):
            if p not in frozen_pipes:
                # 🔴 ПАТЧ: Обережне роздування. Замість вибуху вартості, піднімаємо лише ~35% вільних труб на 1-2 кроки
                if random.random() < 0.35:
                    boost = random.choice([1, 2])
                    fresh[p] = min(self.ctx.max_d_idx, fresh[p] + boost)
                
        squeezed = self.ls.gradient_squeeze(fresh, locked_pipes=frozen_pipes, max_passes=None, quick_mode=False, dyn_bonus=dyn_bonus)
        
        cost, p_val, feas, _ = self.ctx.get_cached_stats(squeezed)
        if not feas or p_val < self.ctx.simulator.config.h_min:
            return None, None, ""
            
        return squeezed, frozen_pipes, f"SEGMENT-RESTART: Froze {len(frozen_pipes)} pipes, partial Top-Down squeeze."

    
    def corridor_search_kick(self, indices, target_sol):
        diff_pipes = [i for i in range(self.ctx.num_pipes) if indices[i] != target_sol[i]]
        if not diff_pipes:
            return None, None, ""
            
        child = list(indices)
        locked = set()
        
        for i in diff_pipes:
            if random.random() < 0.5:
                child[i] = target_sol[i]
                # 🔴 ПАТЧ 4: Блокуємо ТІЛЬКИ ті труби, які дійсно перейняли значення target_sol
                locked.add(i) 
                
        if not locked:
            return None, None, ""
            
        healed, ok, boosts = self.ls.heal_network(child, locked)
        if not ok: return None, None, ""
        
        return healed, locked, f"CORRIDOR: Interpolated {len(diff_pipes)} diff pipes. Healed {boosts}x."
    
    def crossover_with_peer_kick(self, my_sol, peer_sol, my_cost, peer_cost):
        """Схрещування з успішним рішенням іншого воркера"""
        child = []
        total = my_cost + peer_cost
        # 🔴 ПАТЧ (Bug 3): Захист від ZeroDivisionError
        if total <= 0: 
            return None, None, ""
            
        p_mine = peer_cost / total 
        
        locked = set()
        for i, (a, b) in enumerate(zip(my_sol, peer_sol)):
            if a == b:
                child.append(a)
            else:
                chosen = a if random.random() < p_mine else b
                child.append(chosen)
                locked.add(i)
                
        healed, ok, _ = self.ls.heal_network(child, locked)
        if not ok: return None, None, ""
        
        return healed, locked, f"IPC-CROSSOVER: Merged with peer (Mine: {p_mine:.0%}, Peer: {1-p_mine:.0%})."

    def loop_balancing_kick(self, indices, dyn_bonus, failed_pipes=None, current_round=0):
        failed_pipes = failed_pipes or {}
        # 🔴 ПАТЧ (Bug 9): Збільшений tenure і сувора фільтрація
        LOOP_BALANCE_PIPE_TENURE = min(25, max(3, self.ctx.num_pipes // 5))
        
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, "", -1

        try: cycles = nx.cycle_basis(self.ctx.base_G_flow)
        except: return None, None, "", -1
        if not cycles: return None, None, "No cycles found", -1

        best_drop_achieved = -1
        candidates = []

        for cycle_nodes in cycles:
            cycle_indices = []
            full_cycle = cycle_nodes + [cycle_nodes[0]]
            for u, v in zip(full_cycle[:-1], full_cycle[1:]):
                if (u, v) in self.ctx.edge_to_pipe: 
                    cycle_indices.append(self.ctx.edge_to_pipe[(u, v)])

            for candidate_idx in cycle_indices:
                # Відсікаємо труби, які нещодавно були невдалими
                if (current_round - failed_pipes.get(candidate_idx, -999)) < LOOP_BALANCE_PIPE_TENURE:
                    continue
                    
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
                        candidates.append((sq_cost, healed_sol, locked, msg, candidate_idx))
                        best_drop_achieved = max(best_drop_achieved, drop)

        if not candidates: return None, None, "FLOW STEER: Exhaustive search found no valid bypass.", -1
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])
        return chosen[1], chosen[2], chosen[3], chosen[4]

    def ils_perturbation_kick(self, indices, stagnation_counter):
        """ILS: Змінює переважно ті труби, які мають запас тиску (slack)"""
        pct = min(0.35, 0.08 + stagnation_counter * 0.01)
        n_perturb = max(4, int(self.ctx.num_pipes * pct))
        
        # 🔴 ПАТЧ (Bug 8): Менше руйнувань на пізніх стадіях
        is_late = stagnation_counter > 8
        base_limit = 12 if self.ctx.num_pipes >= 200 else 40
        if stagnation_counter > 15:
            hard_limit = int(base_limit * 1.5) # Дозволяємо 18 труб для Балерми!
        else:
            is_late = stagnation_counter > 8
            hard_limit = min(5 if is_late else base_limit, base_limit)
            
        n_perturb = min(n_perturb, hard_limit)
        
        if self.ctx.num_pipes < 200:
            chosen = random.sample(range(self.ctx.num_pipes), n_perturb)
        else:
            unit_losses = self.ctx.get_cached_heuristics(indices)
            pipe_slack = []
            for i in range(self.ctx.num_pipes):
                can_downgrade = indices[i] > 0
                slack_score = (1.0 / (unit_losses[i] + 1e-6)) if can_downgrade else 0.0
                pipe_slack.append((i, slack_score))
                
            # Вибираємо тільки з труб, що мають реальний запас (slack)
            pipe_slack.sort(key=lambda x: x[1], reverse=True)
            n_slack = int(self.ctx.num_pipes * 0.3) 
            slack_pool = [p for p, _ in pipe_slack[:n_slack]]
            tight_pool = [p for p, _ in pipe_slack[n_slack:]]
            
            n_from_slack = max(1, int(n_perturb * 0.7))
            n_from_tight = n_perturb - n_from_slack
            
            chosen = random.sample(slack_pool, min(n_from_slack, len(slack_pool)))
            if n_from_tight > 0 and tight_pool:
                chosen += random.sample(tight_pool, min(n_from_tight, len(tight_pool)))
            
        kicked = list(indices)
        locked = set()
        
        for p_idx in chosen:
            delta = random.choice([-1, 1, 2]) if stagnation_counter > 8 else random.choice([-1, 1])
            new_val = max(0, min(self.ctx.max_d_idx, kicked[p_idx] + delta))
            kicked[p_idx] = new_val
            if delta > 0:
                locked.add(p_idx)
                
        healed, ok, boosts = self.ls.heal_network(kicked, locked)
        if not ok: return None, None, ""
        
        msg = f"ILS-PERTURB (Slack-Aware): Shifted {len(chosen)} pipes. Healed {boosts}x."
        return healed, locked, msg

    def zero_sum_shift_kick(self, indices):
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        unit_losses = self.ctx.get_cached_heuristics(indices)
        
        # 🔴 ФІКС: Менше пар для обміну на великих мережах (захист від деструктивності)
        if self.ctx.num_pipes >= 200:
            max_pairs = 1
        else:
            max_pairs = min(3, max(1, self.ctx.num_pipes // 20))
            
        kicked = list(indices)
        locked = set()
        
        # Кандидати на апгрейд (Високе тертя, але відносно дешево підняти)
        upgrade_candidates = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] < self.ctx.max_d_idx:
                cost_diff = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]+1] - self.ctx.costs_array[kicked[i]])
                score = unit_losses[i] / max(cost_diff, 1.0)
                upgrade_candidates.append((i, cost_diff, score))
                
        # Кандидати на даунгрейд (Низьке тертя, і можна добре зекономити)
        downgrade_candidates = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] > 0:
                cost_diff = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]] - self.ctx.costs_array[kicked[i]-1])
                score = cost_diff / max(unit_losses[i], 1e-5)
                downgrade_candidates.append((i, cost_diff, score))
                
        upgrade_candidates.sort(key=lambda x: x[2], reverse=True)
        downgrade_candidates.sort(key=lambda x: x[2], reverse=True)
        
        exchanges = 0
        used_pipes = set()
        
        for up_pipe, up_cost, _ in upgrade_candidates:
            if exchanges >= max_pairs: break
            if up_pipe in used_pipes: continue
            
            # Шукаємо пару для даунгрейду, яка приблизно компенсує вартість (zero-sum)
            best_down_pipe = -1
            best_diff = float('inf')
            
            for down_pipe, down_cost, _ in downgrade_candidates:
                if down_pipe in used_pipes or down_pipe == up_pipe: continue
                
                cost_balance = abs(up_cost - down_cost)
                if cost_balance < best_diff and cost_balance < (up_cost * 0.5): # Дозволяємо до 50% відхилення ціни
                    best_diff = cost_balance
                    best_down_pipe = down_pipe
                    
            if best_down_pipe != -1:
                kicked[up_pipe] += 1
                kicked[best_down_pipe] -= 1
                locked.add(up_pipe)
                locked.add(best_down_pipe)
                used_pipes.add(up_pipe)
                used_pipes.add(best_down_pipe)
                exchanges += 1

        if not locked: return None, None, ""
        return kicked, locked, f"ZERO-SUM: Exchanged {exchanges} pairs (Cost-Optimized)."
    
    def submarine_oscillation_kick(self, indices):
        """Strategic Oscillation: Радикально ріже магістралі і лікує мережу периферією"""
        unit_losses = self.ctx.get_cached_heuristics(indices)
        
        # Шукаємо "жирні" магістралі (великий діаметр, але відносно низьке падіння тиску на метр)
        fat_mains = []
        for i in range(self.ctx.num_pipes):
            if indices[i] > 3: # Беремо тільки відносно великі труби
                fat_mains.append((i, unit_losses[i]))
                
        # Сортуємо: чим менша втрата тиску, тим безпечніше її різати
        fat_mains.sort(key=lambda x: x[1]) 
        
        if not fat_mains: return None, None, ""
            
        kicked = list(indices)
        locked = set()
        
        # КРОК 1: ЗАНУРЕННЯ. Жорстко ріжемо 3-5 магістралей
        n_cuts = random.choice([3, 4, 5])
        cut_pipes = [x[0] for x in fat_mains[:n_cuts * 2]] # Беремо з топ-кандидатів
        chosen_cuts = random.sample(cut_pipes, min(n_cuts, len(cut_pipes)))
        
        for p_idx in chosen_cuts:
            # Зменшуємо діаметр на 2 кроки відразу (глибоке занурення!)
            kicked[p_idx] = max(0, kicked[p_idx] - 2) 
            # КРОК 2: БЛОКУВАННЯ. Забороняємо лікувати ці труби
            locked.add(p_idx) 
            
        # КРОК 3: СПЛИВАННЯ. Змушуємо мережу лікуватися через інші труби
        healed, ok, boosts = self.ls.heal_network(kicked, locked)
        
        if not ok: return None, None, ""
        
        return healed, locked, f"SUBMARINE: Cut {len(chosen_cuts)} mains (locked). Healed {boosts}x peripheral pipes."