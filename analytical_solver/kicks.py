import random
import networkx as nx
import numpy as np
import itertools

class KickStrategies:
    def __init__(self, ctx, local_search):
        self.ctx = ctx
        self.ls = local_search

    # ==========================================
    # 1. ТОПОЛОГІЧНІ ТА ГІДРАВЛІЧНІ (Еліта)
    # ==========================================
    def forcing_hand_kick(self, indices, T, **kwargs):
        """Атака на критичний вузол. Агресія масштабується від T."""
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return indices, set(), ""

        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return indices, set(), ""
            
        kicked, locked = list(indices), set()
        
        # Агресія: від 5% (холодний) до 40% (гарячий) критичного шляху
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
        """Виправлення горлечок. Температура визначає глибину пошуку."""
        failed_pipes = kwargs.get('failed_pipes', {})
        current_round = kwargs.get('current_round', 0)
        
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return indices, set(), "", -1
        
        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes: return indices, set(), "", -1
        
        kicked, locked = list(indices), set()
        failed_id = -1
        
        for i in range(len(path_pipes) - 1, 0, -1):
            curr_p, prev_p = path_pipes[i], path_pipes[i-1]
            if curr_p in failed_pipes and (current_round - failed_pipes[curr_p]) < 10: continue
                
            if indices[curr_p] < indices[prev_p]:
                if kicked[curr_p] < self.ctx.max_d_idx:
                    kicked[curr_p] += 1
                    locked.add(curr_p)
                    return kicked, locked, f"BOTTLENECK: Boosted Pipe {curr_p} (Taper Violation)", curr_p
                    
        # Якщо горлечок немає, беремо найгіршу трубу з першої половини шляху
        unit_losses = self.ctx.get_cached_heuristics(indices)
        search_depth = max(1, int(len(path_pipes) * (0.3 + 0.4 * T))) # Чим вище T, тим глибше шукаємо
        
        candidates = [(idx, unit_losses[idx]) for idx in path_pipes[:search_depth] 
                      if indices[idx] < self.ctx.max_d_idx 
                      and not (idx in failed_pipes and (current_round - failed_pipes[idx]) < 10)]
        
        if not candidates: return indices, set(), "", -1
        best_pipe = max(candidates, key=lambda x: x[1])[0]
        kicked[best_pipe] += 1
        locked.add(best_pipe)
        return kicked, locked, f"BOTTLENECK: Max-Loss Pipe {best_pipe} boosted.", best_pipe

    def topological_inversion_kick(self, indices, T, **kwargs):
        """Шукає альтернативні маршрути."""
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
        
        # Агресія пересадки маршруту залежить від T
        aggressiveness = 0.10 + (0.50 * T)
        max_pipes = max(2, int(len(best_alt['indices']) * aggressiveness))
        
        indices_to_change = best_alt['indices']
        if len(indices_to_change) > max_pipes:
             indices_to_change = random.sample(indices_to_change, max_pipes)

        for idx in indices_to_change:
            force_idx = min(max(target_capacity_idx, self.ctx.max_d_idx - 1), self.ctx.max_d_idx)
            if kicked[idx] < force_idx:
                kicked[idx] = force_idx
                locked.add(idx)
            elif kicked[idx] == force_idx: locked.add(idx)

        return kicked, locked, f"TOPO-INV: Boosted Alt Path (Overlap {best_alt['overlap']:.2f}, T={T:.2f})", best_alt['signature']

    def loop_balancing_kick(self, indices, T, **kwargs):
        """Балансування циклів."""
        failed_pipes = kwargs.get('failed_pipes', {})
        current_round = kwargs.get('current_round', 0)
        dyn_bonus = kwargs.get('dyn_bonus', 0)
        
        LOOP_BALANCE_PIPE_TENURE = min(25, max(3, self.ctx.num_pipes // 5))
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, "", -1

        if not hasattr(self, '_cached_cycles'):
            try: self._cached_cycles = nx.cycle_basis(self.ctx.base_G_flow)
            except: self._cached_cycles = []
        cycles = self._cached_cycles
        
        if not cycles: return None, None, "No cycles found", -1

        # Максимальне падіння діаметра залежить від температури
        max_allowed_drop = 1 + int(2 * T) 

        best_drop_achieved = -1
        candidates = []
        random.shuffle(cycles)

        for cycle_nodes in cycles:
            cycle_indices = []
            full_cycle = cycle_nodes + [cycle_nodes[0]]
            for u, v in zip(full_cycle[:-1], full_cycle[1:]):
                if (u, v) in self.ctx.edge_to_pipe: cycle_indices.append(self.ctx.edge_to_pipe[(u, v)])

            random.shuffle(cycle_indices)
            for candidate_idx in cycle_indices:
                if (current_round - failed_pipes.get(candidate_idx, -999)) < LOOP_BALANCE_PIPE_TENURE: continue
                curr_d_idx = indices[candidate_idx]
                if curr_d_idx < 2: continue 
                
                max_drop = min(max_allowed_drop, curr_d_idx) 
                for drop in range(max_drop, 0, -1):
                    if drop < best_drop_achieved: continue
                    kicked, locked = list(indices), set()
                    kicked[candidate_idx] -= drop
                    locked.add(candidate_idx)
                    
                    healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
                    if is_feasible:
                        test_squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
                        sq_cost, _, _, _ = self.ctx.get_cached_stats(test_squeezed)
                        candidates.append((sq_cost, healed_sol, locked, f"FLOW STEER: Cut Pipe {candidate_idx + 1} (-{drop}). Healed {boosts}x.", candidate_idx))
                        best_drop_achieved = max(best_drop_achieved, drop)
            
            if len(candidates) >= 5: break

        if not candidates: return None, None, "FLOW STEER: Exhaustive search found no valid bypass.", -1
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])
        return chosen[1], chosen[2], chosen[3], chosen[4]

    # ==========================================
    # 2. ХІРУРГІЯ (Точна оптимізація вартості)
    # ==========================================
    # def zero_sum_shift_kick(self, indices, T, **kwargs):
    #     """Ендшпіль: Шукає ідеальний обмін ємності між магістраллю та периферією."""
    #     _, _, _, crit_node = self.ctx.get_cached_stats(indices)
    #     if not crit_node or crit_node == "ERR": return None, None, ""

    #     unit_losses = self.ctx.get_cached_heuristics(indices)
        
    #     # Чим вище T, тим більше пар ми обмінюємо одночасно. Але якщо T мале (Ендшпіль) - шукаємо 1 ідеальну пару
    #     max_pairs_to_change = max(1, int((self.ctx.num_pipes * 0.03) * T))
            
    #     kicked, locked = list(indices), set()
    #     upgrade_candidates, downgrade_candidates = [], []
        
    #     for i in range(self.ctx.num_pipes):
    #         if kicked[i] < self.ctx.max_d_idx:
    #             c_diff = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]+1] - self.ctx.costs_array[kicked[i]])
    #             upgrade_candidates.append((i, c_diff, unit_losses[i] / max(c_diff, 1.0)))
    #         if kicked[i] > 0:
    #             c_diff = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]] - self.ctx.costs_array[kicked[i]-1])
    #             downgrade_candidates.append((i, c_diff, c_diff / max(unit_losses[i], 1e-5)))
                
    #     upgrade_candidates.sort(key=lambda x: x[2], reverse=True)
    #     downgrade_candidates.sort(key=lambda x: x[2], reverse=True)
        
    #     # 🔴 РОЗШИРЕННЯ ДЛЯ ЕНДШПІЛЮ: Створюємо кілька варіантів і тестуємо їх
    #     best_exchange_sol = None
    #     best_exchange_cost = float('inf')
    #     best_exchange_locked = set()
        
    #     # Перевіряємо топ-5 найперспективніших обмінів
    #     test_limit = 5 if T < 0.3 else 1
    #     valid_exchanges_found = 0
        
    #     for up_pipe, up_cost, _ in upgrade_candidates[:15]:
    #         if valid_exchanges_found >= test_limit: break
            
    #         for down_pipe, down_cost, _ in downgrade_candidates[:15]:
    #             if down_pipe == up_pipe: continue
                
    #             # Допускаємо обмін, якщо збільшення труби не сильно дорожче за зменшення іншої
    #             if up_cost < (down_cost * 1.5):
    #                 test_sol = list(indices)
    #                 test_sol[up_pipe] += 1
    #                 test_sol[down_pipe] -= 1
    #                 test_locked = {up_pipe, down_pipe}
                    
    #                 # Швидка перевірка чи не впав тиск
    #                 healed_sol, is_feas, _ = self.ls.heal_network(test_sol, test_locked)
    #                 if is_feas:
    #                     c, _, _, _ = self.ctx.get_cached_stats(healed_sol)
    #                     if c < best_exchange_cost:
    #                         best_exchange_cost = c
    #                         best_exchange_sol = healed_sol
    #                         best_exchange_locked = test_locked
    #                     valid_exchanges_found += 1
    #                     break # Переходимо до наступного up_pipe
                        
    #     if best_exchange_sol:
    #         return best_exchange_sol, best_exchange_locked, f"ZERO-SUM (Deep): Exchanged pairs for cost {best_exchange_cost/1e6:.4f}M$."
            
    #     return None, None, ""

    def zero_sum_shift_kick(self, indices, T, **kwargs):
        """
        Універсальний Zero-Sum Обмін (1-to-1 або 1-to-N).
        Знаходить 1 трубу для розширення (інвестиція) і N труб для звуження (економія).
        Кількість N (асиметрія) адаптивно залежить від Температури.
        """
        _, _, feas, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        unit_losses = self.ctx.get_cached_heuristics(indices)
        kicked = list(indices)
        
        # 1. Формуємо пул Інвестицій (Upgrades: високий опір, низька ціна)
        upgrades = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] < self.ctx.max_d_idx:
                c_up = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]+1] - self.ctx.costs_array[kicked[i]])
                upgrades.append((i, c_up, unit_losses[i] / max(c_up, 1.0)))
        upgrades.sort(key=lambda x: x[2], reverse=True)

        # 2. Формуємо пул Економії (Downgrades: низький опір, висока ціна)
        downgrades = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] > 0:
                c_down = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]] - self.ctx.costs_array[kicked[i]-1])
                downgrades.append((i, c_down, c_down / max(unit_losses[i], 1e-5)))
        downgrades.sort(key=lambda x: x[2], reverse=True)

        best_sol, best_locked, best_cost = None, None, float('inf')
        
        # Адаптивні параметри на основі Температури
        tests_limit = 5 if T < 0.3 else 2
        # Чим вища T, тим більше периферійних труб дозволено зрізати за 1 інвестицію
        max_downgrades = 1 if T < 0.2 else (3 if T < 0.5 else 6)
        
        valid_found = 0

        # 3. Пошук ідеальної комбінації
        for up_idx, cost_invest, _ in upgrades[:15]:
            if valid_found >= tests_limit: break
            
            test_sol = list(indices)
            test_sol[up_idx] += 1 # Робимо інвестицію
            locked_set = {up_idx}
            savings = 0
            down_count = 0
            
            for down_idx, cost_save, _ in downgrades:
                if down_idx == up_idx or down_idx in locked_set: continue
                
                test_sol[down_idx] -= 1 # Забираємо діаметр
                locked_set.add(down_idx)
                savings += cost_save
                down_count += 1
                
                # Як тільки вийшли в чистий прибуток
                if savings > cost_invest:
                    # Перевіряємо гідравлічну валідність
                    healed_sol, ok, _ = self.ls.heal_network(test_sol, locked_set)
                    if ok:
                        c, _, _, _ = self.ctx.get_cached_stats(healed_sol)
                        if c < best_cost:
                            best_cost, best_sol, best_locked = c, healed_sol, locked_set
                        valid_found += 1
                    break # Переходимо до наступної інвестиції (up_idx)
                    
                # Якщо ліміт асиметрії вичерпано, а прибутку ще немає - відкидаємо комбінацію
                if down_count >= max_downgrades:
                    break 

        if best_sol:
            mode_str = "1-to-1" if len(best_locked) == 2 else f"1-to-{len(best_locked)-1}"
            return best_sol, best_locked, f"ZERO-SUM ({mode_str}): P{list(best_locked)[0]} upgraded. Cost: {best_cost/1e6:.4f}M$"
            
        return None, None, ""

    def peripheral_trim_kick(self, indices, T, **kwargs):
        """Об'єднаний FINISHER та SYNC_TRIM. Відкушує тупикові гілки."""
        indices_copy, periphery_pipes = self.ctx.get_lazy_periphery(indices)
        if not periphery_pipes: return None, None, ""
        
        # При T=0 відкушуємо 1 трубу. При T=1 відкушуємо до 3 труб одночасно.
        pipes_to_cut = 1 + int(2 * T)
        
        candidates = []
        combo_limit = min(50, max(5, self.ctx.num_pipes // 4)) 
        
        # Якщо треба обрізати 1 трубу - перебираємо по одній. Інакше - комбінаціями.
        if pipes_to_cut == 1:
            all_combos = [(p,) for p in periphery_pipes[:combo_limit]]
        else:
            all_combos = list(itertools.combinations(periphery_pipes[:combo_limit], pipes_to_cut))
            random.shuffle(all_combos)
        
        max_checks = min(25, max(5, self.ctx.num_pipes // 10))
        dyn_bonus = kwargs.get('dyn_bonus', 0)
        
        for combo in all_combos[:max_checks]:
            if any(indices_copy[p] == 0 for p in combo): continue
            
            test_sol = list(indices_copy)
            for p in combo: test_sol[p] -= 1
            locked = set(combo)
            
            healed_sol, is_feasible, boosts = self.ls.heal_network(test_sol, locked)
            if is_feasible:
                h_c, h_p, _, _ = self.ctx.get_cached_stats(healed_sol) 
                h_score = h_c - ((h_p - self.ctx.simulator.config.h_min) * dyn_bonus)
                candidates.append((h_score, healed_sol, locked, boosts))
                
        if not candidates: return None, None, ""
        
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3]) # Беремо один з найкращих
        return chosen[1], chosen[2], f"TRIM: Shrunk {len(chosen[2])} peripheral pipes. Healed {chosen[3]}x."

    # ==========================================
    # 3. ВИХІД З ЯМИ (Heavy Mutations)
    # ==========================================
    def smart_perturbation_kick(self, indices, T, **kwargs):        
        """(Об'єднаний ILS/VNS) Рандомізує певну кількість труб."""
        # Агресія: від 2% до 20% труб
        pct = 0.02 + (0.18 * T)
        n_perturb = max(1, int(self.ctx.num_pipes * pct))
        
        kicked, locked = list(indices), set()
        candidates = [i for i in range(self.ctx.num_pipes) if 0 < kicked[i] < self.ctx.max_d_idx]
        if len(candidates) < n_perturb: candidates = list(range(self.ctx.num_pipes))
            
        chosen = random.sample(candidates, min(n_perturb, len(candidates)))
        
        max_delta = 2 if T > 0.6 else 1
        
        for p_idx in chosen:
            # Більша ймовірність легких зсувів, менша - екстремальних
            if max_delta > 1: delta = random.choices([-2, -1, 1, 2], weights=[1, 3, 3, 1])[0]
            else: delta = random.choice([-1, 1])
                
            new_val = max(0, min(self.ctx.max_d_idx, kicked[p_idx] + delta))
            kicked[p_idx] = new_val
            if delta > 0: locked.add(p_idx) 
                
        healed, ok, boosts = self.ls.heal_network(kicked, locked)
        if not ok: return None, None, ""
        return healed, locked, f"SMART-PERTURB: Shifted {len(chosen)} pipes (T={T:.2f}). Healed {boosts}x."

    def ruin_and_recreate_kick(self, indices, T, **kwargs):
        """Просторове руйнування кластеру."""
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        ruin_center = crit_node if T < 0.8 else random.choice(list(self.ctx.base_G_flow.nodes()))
        
        # Радіус руйнування залежить від T
        target_pct = 0.02 + (0.13 * T) 
        target_pipes = max(3, int(self.ctx.num_pipes * target_pct))
        
        pipe_with_dist, seen_pipes = [], set()
        try:
            for node, dist in nx.single_source_shortest_path_length(self.ctx.base_G_flow, ruin_center, cutoff=10).items():
                for p in self.ctx.node_to_pipes.get(node, []):
                    if p not in seen_pipes:
                        seen_pipes.add(p)
                        pipe_with_dist.append((dist, p))
        except: pass

        if not pipe_with_dist: return None, None, ""
        pipe_with_dist.sort(key=lambda x: x[0])
        cluster_pipes = set(p for _, p in pipe_with_dist[:target_pipes])

        kicked, ruined_count = list(indices), 0
        max_drop = 1 + int(2 * T) # Падіння діаметра: від 1 до 3
        
        for p in cluster_pipes:
            if kicked[p] > 0:
                kicked[p] = max(0, kicked[p] - random.randint(1, max_drop))
                ruined_count += 1

        if ruined_count == 0: return None, None, ""

        healed_sol, is_feas, boosts = self.ls.heal_network(kicked, set())
        if not is_feas: return None, None, ""
        return healed_sol, set(), f"R&R: Downgraded {ruined_count} clustered pipes (Drop up to {max_drop}). Rebuilt {boosts}x."
    
    def basin_escape(self, indices, T, **kwargs):
        """Кросовер з глобальним архівом."""
        global_archive = kwargs.get('global_archive', [])
        if not global_archive or len(global_archive) < 2: return None, None, "Archive too small"

        best_dist, diverse_sol = -1, None
        for _, arch_sol in global_archive:
            dist = sum(1 for a, b in zip(indices, arch_sol) if a != b)
            if dist > best_dist: best_dist, diverse_sol = dist, arch_sol

        if diverse_sol is None or best_dist < max(2, int(self.ctx.num_pipes * 0.03)):
            return None, None, "No diverse target"

        diff_pipes = [i for i in range(self.ctx.num_pipes) if indices[i] != diverse_sol[i]]
        
        # Кількість труб для пересадки залежить від T
        n_replace = max(1, int(len(diff_pipes) * (0.1 + 0.4 * T)))
        if n_replace < 1: return None, None, ""
        
        replace_pipes = random.sample(diff_pipes, n_replace)
        kicked = list(indices)
        for p in replace_pipes: kicked[p] = diverse_sol[p]

        healed, ok, boosts = self.ls.heal_network(kicked, set())
        if not ok: return None, None, "Heal failed"
        return healed, set(), f"BASIN-ESCAPE: Transplanted {n_replace} pipes from archive. Healed {boosts}x."
    
    def spatial_perturb_kick(self, indices, T, **kwargs):
        """
        Топологічна мікро-хірургія. Мутує труби лише у вибраному районі мережі.
        """
        import random
        import networkx as nx
        
        if not hasattr(self.ctx, 'base_G_flow'):
            return self.smart_perturbation_kick(indices, T, **kwargs)

        G = self.ctx.base_G_flow
        if not G.nodes: return None, None, ""

        radius = 1 + int(4.0 * T)
        epicenter = random.choice(list(G.nodes()))
        local_nodes = set(nx.single_source_shortest_path_length(G, epicenter, cutoff=radius).keys())
        
        local_pipes = set()
        
        # 🔴 РОЗУМНИЙ ОБХІД РЕБЕР
        if G.is_multigraph():
            for u, v, k in G.edges(keys=True):
                if (u in local_nodes or v in local_nodes) and isinstance(k, int):
                    local_pipes.add(k)
        else:
            for u, v, data in G.edges(data=True):
                if u in local_nodes or v in local_nodes:
                    for val in data.values():
                        if isinstance(val, int) and 0 <= val < self.ctx.num_pipes:
                            local_pipes.add(val)
                            break
                            
        # Якщо не вдалося витягнути індекси, використовуємо SMART_PERTURB як запасний варіант
        if not local_pipes: 
            return self.smart_perturbation_kick(indices, T, **kwargs)

        kicked = list(indices)
        locked = set()
        
        for i in range(self.ctx.num_pipes):
            if i not in local_pipes:
                locked.add(i)
                
        mutations_made = 0
        intensity = 0.1 + (0.3 * T) 
        target_mutations = max(1, int(len(local_pipes) * intensity))
        
        for p_idx in random.sample(list(local_pipes), min(len(local_pipes), target_mutations)):
            shift = random.choice([-2, -1, 1, 2]) if T > 0.6 else random.choice([-1, 1])
            new_val = kicked[p_idx] + shift
            if 0 <= new_val <= self.ctx.max_d_idx:
                kicked[p_idx] = new_val
                mutations_made += 1

        return kicked, locked, f"SPATIAL-PERTURB: Epicenter {epicenter} (R={radius}). Mutated {mutations_made} pipes. Frozen {len(locked)}."
    
# import random
# import networkx as nx
# import numpy as np
# import itertools

# class KickStrategies:
#     def __init__(self, ctx, local_search):
#         self.ctx = ctx
#         self.ls = local_search
#         self.n = ctx.num_pipes
#         self.is_small = self.n < 50
#         self.is_large = self.n >= 200
        
#     def _get_aggressiveness(self):
#         r = random.random()
#         if r < 0.60: return 0.05 
#         elif r < 0.90: return 0.15
#         else: return 0.30

#     def forcing_hand_kick(self, indices):
#         _, _, _, crit_node = self.ctx.get_cached_stats(indices)
#         if not crit_node or crit_node == "ERR": return indices, set(), ""

#         path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
#         if not path_pipes: return indices, set(), ""
            
#         kicked, locked = list(indices), set()
        
#         agg = self._get_aggressiveness()
#         if self.is_small: 
#             agg *= 0.5
            
#         limit = max(1, int(len(path_pipes) * agg))
#         unit_losses = self.ctx.get_cached_heuristics(indices)
        
#         worst_pipes = sorted(path_pipes, key=lambda i: unit_losses[i], reverse=True)[:limit]
        
#         for idx in worst_pipes:
#             if kicked[idx] < self.ctx.max_d_idx:
#                 kicked[idx] += 1
#                 locked.add(idx) 
        
#         healed = self.ls.gradient_squeeze(kicked, locked_pipes=locked, max_passes=3, quick_mode=True)
#         return healed, locked, f"FORCING HAND: Forced {len(worst_pipes)} pipes (Agg: {agg:.2f})"

#     def upstream_bottleneck_kick(self, indices, failed_pipes=None, current_round=0):
#         failed_pipes = failed_pipes or {}
#         _, _, _, crit_node = self.ctx.get_cached_stats(indices)
#         if not crit_node or crit_node == "ERR": return indices, set(), "", -1
        
#         path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
#         if not path_pipes: return indices, set(), "", -1
        
#         kicked, locked = list(indices), set()
#         bottleneck_found = False
#         failed_id = -1
#         d_prev, d_curr = -1, -1
        
#         for i in range(len(path_pipes) - 1, 0, -1):
#             curr_p = path_pipes[i]
#             prev_p = path_pipes[i-1]
            
#             if curr_p in failed_pipes and (current_round - failed_pipes[curr_p]) < 10:
#                 continue
                
#             d_curr = indices[curr_p]
#             d_prev = indices[prev_p]
            
#             if d_curr < d_prev:
#                 if kicked[curr_p] < self.ctx.max_d_idx:
#                     kicked[curr_p] += 1
#                     locked.add(curr_p)
#                     bottleneck_found = True
#                     failed_id = curr_p
#                     break
                    
#         if not bottleneck_found:
#             unit_losses = self.ctx.get_cached_heuristics(indices)
#             half = max(1, len(path_pipes) // 2)
#             candidates = [(idx, unit_losses[idx]) for idx in path_pipes[:half] 
#                           if indices[idx] < self.ctx.max_d_idx 
#                           and not (idx in failed_pipes and (current_round - failed_pipes[idx]) < 10)]
            
#             if not candidates:
#                 return indices, set(), "", -1
                
#             best_pipe = max(candidates, key=lambda x: x[1])[0]
#             kicked[best_pipe] = min(self.ctx.max_d_idx, kicked[best_pipe] + 1)
#             locked.add(best_pipe)
#             failed_id = best_pipe
#             return kicked, locked, f"BOTTLENECK: Max-Loss Pipe {best_pipe} boosted.", failed_id

#         return kicked, locked, f"BOTTLENECK: Boosted Pipe {failed_id} (Taper Violation (Idx{d_prev}->Idx{d_curr}))", failed_id

#     def topological_inversion_kick(self, indices, tabu_set):
#         _, _, _, crit_node = self.ctx.get_cached_stats(indices)
#         if not crit_node or crit_node == "ERR": return None, None, "", None
        
#         dom_pipes, dom_nodes = self.ctx.get_dominant_path(indices, crit_node)
#         if not dom_nodes: return None, None, "", None
        
#         source = dom_nodes[0]
#         dom_edges = set(tuple(sorted((u, v))) for u, v in zip(dom_nodes[:-1], dom_nodes[1:]))
        
#         target_capacity_idx = int(np.mean([indices[p] for p in dom_pipes])) if dom_pipes else self.ctx.max_d_idx
        
#         temp_G = self.ctx.base_G_flow.copy()
#         for u, v in temp_G.edges():
#             idx = self.ctx.edge_to_pipe[(u, v)]
#             temp_G[u][v]['weight'] = 100.0 / (indices[idx] + 1)
            
#         candidates = []
#         for _ in range(15):
#             try:
#                 path = nx.shortest_path(temp_G, source, crit_node, weight='weight')
#                 if path != dom_nodes:
#                     candidates.append(path)
#                 for u, v in zip(path[:-1], path[1:]):
#                     temp_G[u][v]['weight'] *= 5.0
#             except nx.NetworkXNoPath:
#                 break

#         scored_paths = []
#         for p_nodes in candidates:
#             p_indices, overlap, length = [], 0, 0
#             for u, v in zip(p_nodes[:-1], p_nodes[1:]):
#                 edge_key = tuple(sorted((u, v)))
#                 if edge_key in dom_edges: overlap += 1
#                 if (u, v) in self.ctx.edge_to_pipe:
#                     p_indices.append(self.ctx.edge_to_pipe[(u, v)])
#                     length += 1
            
#             if length == 0: continue
#             overlap_ratio = overlap / length
#             signature = tuple(sorted(p_indices))
#             if signature in tabu_set: continue
#             scored_paths.append({'indices': p_indices, 'overlap': overlap_ratio, 'signature': signature})

#         scored_paths.sort(key=lambda x: x['overlap'])
#         if not scored_paths: return None, None, "All paths tabu", None

#         best_alt = scored_paths[0]
#         kicked, locked, push_count = list(indices), set(), 0
        
#         c_current, _, _, _ = self.ctx.get_cached_stats(indices)
#         aggressiveness = 0.4 if c_current > 2_200_000 else (0.2 if c_current > 2_050_000 else 0.08)
#         max_pipes_to_change = max(2, int(len(best_alt['indices']) * aggressiveness))
        
#         indices_to_change = best_alt['indices']
#         if len(indices_to_change) > max_pipes_to_change:
#              indices_to_change = random.sample(indices_to_change, max_pipes_to_change)

#         for idx in indices_to_change:
#             force_idx = max(target_capacity_idx, self.ctx.max_d_idx - 1) 
#             force_idx = min(force_idx, self.ctx.max_d_idx)
#             if kicked[idx] < force_idx:
#                 kicked[idx] = force_idx
#                 locked.add(idx)
#                 push_count += 1
#             elif kicked[idx] == force_idx:
#                  locked.add(idx)

#         msg = f"Topo-Kick (Dijkstra): Boosting Alt Path (Overlap {best_alt['overlap']:.2f}, {push_count} boosted)"
#         return kicked, locked, msg, best_alt['signature']

#     def micro_trim_kick(self, indices):
#         indices_copy, periphery_pipes = self.ctx.get_lazy_periphery(indices)
#         if not periphery_pipes: return None, None, ""
        
#         kicked, locked, shrunk_count = list(indices_copy), set(), 0
#         limit = random.choice([1, 2, 3])
        
#         for idx in periphery_pipes:
#             if kicked[idx] > 0: 
#                 kicked[idx] -= 1
#                 locked.add(idx)
#                 shrunk_count += 1
#             if shrunk_count >= limit: break
                
#         if shrunk_count > 0:
#             healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
#             if is_feasible:
#                 return healed_sol, locked, f"FINISHER: Force-shrunk {shrunk_count} periphery pipes. Healed {boosts}x."
#         return None, None, ""
    
#     def sync_trim_kick(self, indices, dyn_bonus):
#         indices_copy, periphery_pipes = self.ctx.get_lazy_periphery(indices)
#         if not periphery_pipes: return None, None, ""
#         candidates = []
        
#         combo_limit = min(100, max(10, self.ctx.num_pipes // 3)) 
#         all_pairs = list(itertools.combinations(periphery_pipes[:combo_limit], 2))
#         random.shuffle(all_pairs)
        
#         max_checks = min(500, max(60, self.ctx.num_pipes))
        
#         for d1, d2 in all_pairs[:max_checks]:
#             if indices_copy[d1] == 0 or indices_copy[d2] == 0: continue
            
#             test_sol = list(indices_copy)
#             test_sol[d1] -= 1
#             test_sol[d2] -= 1
#             locked = {d1, d2}
            
#             healed_sol, is_feasible, boosts = self.ls.heal_network(test_sol, locked)
#             if is_feasible:
#                 h_c, h_p, _, _ = self.ctx.get_cached_stats(healed_sol) 
#                 h_surplus = h_p - self.ctx.simulator.config.h_min
#                 h_score = h_c - (h_surplus * dyn_bonus)
#                 candidates.append((h_score, healed_sol, locked, boosts))
                
#         if not candidates: return None, None, ""
#         candidates.sort(key=lambda x: x[0])
#         validated_candidates = []
        
#         for _, healed_sol, locked, boosts in candidates[:5]:
#             squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
#             sq_c, _, _, _ = self.ctx.get_cached_stats(squeezed)
#             validated_candidates.append((sq_c, healed_sol, locked, boosts))
                
#         if not validated_candidates: return None, None, ""
#         validated_candidates.sort(key=lambda x: x[0])
#         chosen = random.choice(validated_candidates[:3])
#         _, best_final_sol, best_locked, best_boosts = chosen
        
#         return best_final_sol, best_locked, f"SYNC-TRIM: Shrunk Pipes {[p+1 for p in best_locked]}. Healed {best_boosts}x."

#     def diameter_diversity_kick(self, indices, stagnation_counter):
#         kicked, locked = list(indices), set()
        
#         base_n = max(5, int(self.ctx.num_pipes * 0.12))
#         aggression_step = max(0.5, self.ctx.num_pipes / 100.0) 
#         n_to_change = min(self.ctx.num_pipes // 2, base_n + int(stagnation_counter * aggression_step))
        
#         pipes_to_change = random.sample(range(self.ctx.num_pipes), n_to_change)
        
#         from collections import Counter
#         diam_counts = Counter(indices)
#         most_common_diam = diam_counts.most_common(1)[0][0]
        
#         for p in pipes_to_change:
#             current_d = kicked[p]
#             if current_d == most_common_diam:
#                 if random.random() < 0.5 and current_d < self.ctx.max_d_idx:
#                     kicked[p] += 1
#                 elif current_d > 0:
#                     kicked[p] -= 1
#             else:
#                 kicked[p] = most_common_diam
#             locked.add(p)
            
#         healed_sol, is_feas, _ = self.ls.heal_network(kicked, locked)
#         if not is_feas: return None, None, ""
        
#         return healed_sol, locked, f"DIAM-DIVERSITY: Changed {len(locked)} pipes to break homogeneity."
    
#     def ruin_and_recreate_kick(self, indices, stagnation_counter):
#         _, _, _, crit_node = self.ctx.get_cached_stats(indices)
#         if not crit_node or crit_node == "ERR": return None, None, ""

#         if stagnation_counter > 8:
#             all_nodes = list(self.ctx.base_G_flow.nodes())
#             ruin_center = random.choice(all_nodes)
#         else:
#             ruin_center = crit_node

#         progress = getattr(self.ctx, 'progress_ratio', 0.0)
#         is_late_game = progress > 0.6
        
#         if is_late_game:
#             max_pct = 0.05
#             base_pct = random.uniform(0.01, 0.02)
#             max_pipes = 10
#         else:
#             max_pct = 0.20 if self.ctx.num_pipes <= 50 else 0.40
#             base_pct = random.uniform(0.10, max_pct / 2)
#             max_pipes = 35

#         target_pct = min(max_pct, base_pct + (stagnation_counter * 0.015)) 
#         target_pipes = max(3, int(self.ctx.num_pipes * target_pct))
#         target_pipes = min(max_pipes, target_pipes)
        
#         pipe_with_dist = []
#         seen_pipes = set()
        
#         try:
#             for node, dist in nx.single_source_shortest_path_length(self.ctx.base_G_flow, ruin_center, cutoff=15).items():
#                 for p in self.ctx.node_to_pipes.get(node, []):
#                     if p not in seen_pipes:
#                         seen_pipes.add(p)
#                         pipe_with_dist.append((dist, p))
#         except:
#             pass

#         if not pipe_with_dist: return None, None, ""
        
#         pipe_with_dist.sort(key=lambda x: x[0])
#         cluster_pipes = set(p for _, p in pipe_with_dist[:target_pipes])

#         kicked = list(indices)
#         ruined_count = 0
        
#         for p in cluster_pipes:
#             if kicked[p] > 0:
#                 if is_late_game:
#                     kicked[p] = max(0, kicked[p] - random.choice([1, 2]))
#                 else:
#                     kicked[p] = 0
#                 ruined_count += 1

#         if ruined_count == 0: return None, None, ""

#         healed_sol, is_feas, boosts = self.ls.heal_network(kicked, set())
#         if not is_feas: return None, None, ""

#         actual_radius = max((d for d, _ in pipe_with_dist[:target_pipes]), default=0)
        
#         return healed_sol, set(), f"R&R (LNS): Ruined {ruined_count} pipes (~{target_pct:.0%} net, Rad: {actual_radius}). Rebuilt {boosts}x."
    
#     def vns_structured_kick(self, indices, stagnation_level):
#         n = self.ctx.num_pipes
#         progress = getattr(self.ctx, 'progress_ratio', 0.0)
#         is_late = progress > 0.6 or stagnation_level > 8
        
#         neighborhood_sizes = [0.05, 0.08, 0.12, 0.15, 0.20]
#         k_idx = min(stagnation_level // 3, len(neighborhood_sizes) - 1)
#         pct = neighborhood_sizes[k_idx]
        
#         hard_limit = 20 if is_late else 35
#         n_change = min(max(3, int(n * pct)), hard_limit)
        
#         unit_losses = self.ctx.get_cached_heuristics(indices)
#         worst_pipes = sorted(range(n), key=lambda i: unit_losses[i], reverse=True)[:n_change + max(5, n//10)]
#         target_pipes = random.sample(worst_pipes, n_change)
        
#         kicked = list(indices)
#         locked = set()
        
#         jump_size = 1 if is_late else 2
        
#         for p in target_pipes:
#             direction = random.choice([-jump_size, jump_size])
#             kicked[p] = max(0, min(self.ctx.max_d_idx, kicked[p] + direction))
            
#             if direction > 0:
#                 locked.add(p)
            
#         healed, ok, boosts = self.ls.heal_network(kicked, locked)
#         if not ok: return None, None, ""
        
#         return healed, locked, f"VNS-KICK (Level {k_idx}): Shifted {n_change} high-loss pipes by ±{jump_size}. Healed {boosts}x."

#     def segment_restart_kick(self, indices, dyn_bonus):
#         n = self.ctx.num_pipes
        
#         freeze_pct = 0.50 if n <= 50 else 0.20
#         n_freeze = max(5, int(n * freeze_pct)) 
        
#         top_candidates = self.ls.get_high_impact_pipes(indices, n_freeze + max(3, n // 10))
#         frozen_pipes = set(random.sample(top_candidates, min(n_freeze, len(top_candidates))))
        
#         fresh = list(indices)
#         for p in range(n):
#             if p not in frozen_pipes:
#                 if random.random() < 0.35:
#                     boost = random.choice([1, 2])
#                     fresh[p] = min(self.ctx.max_d_idx, fresh[p] + boost)
                
#         squeezed = self.ls.gradient_squeeze(fresh, locked_pipes=frozen_pipes, max_passes=None, quick_mode=False, dyn_bonus=dyn_bonus)
        
#         cost, p_val, feas, _ = self.ctx.get_cached_stats(squeezed)
#         if not feas or p_val < self.ctx.simulator.config.h_min:
#             return None, None, ""
            
#         return squeezed, frozen_pipes, f"SEGMENT-RESTART: Froze {len(frozen_pipes)} pipes, partial Top-Down squeeze."

    
#     def corridor_search_kick(self, indices, target_sol):
#         diff_pipes = [i for i in range(self.ctx.num_pipes) if indices[i] != target_sol[i]]
#         if not diff_pipes:
#             return None, None, ""
            
#         child = list(indices)
#         locked = set()
        
#         for i in diff_pipes:
#             if random.random() < 0.5:
#                 child[i] = target_sol[i]
#                 locked.add(i) 
                
#         if not locked:
#             return None, None, ""
            
#         healed, ok, boosts = self.ls.heal_network(child, locked)
#         if not ok: return None, None, ""
        
#         return healed, locked, f"CORRIDOR: Interpolated {len(diff_pipes)} diff pipes. Healed {boosts}x."
    
#     def crossover_with_peer_kick(self, my_sol, peer_sol, my_cost, peer_cost):
#         child = []
#         total = my_cost + peer_cost
#         if total <= 0: 
#             return None, None, ""
            
#         p_mine = peer_cost / total 
        
#         locked = set()
#         for i, (a, b) in enumerate(zip(my_sol, peer_sol)):
#             if a == b:
#                 child.append(a)
#             else:
#                 chosen = a if random.random() < p_mine else b
#                 child.append(chosen)
#                 locked.add(i)
                
#         healed, ok, _ = self.ls.heal_network(child, locked)
#         if not ok: return None, None, ""
        
#         return healed, locked, f"IPC-CROSSOVER: Merged with peer (Mine: {p_mine:.0%}, Peer: {1-p_mine:.0%})."

#     def loop_balancing_kick(self, indices, dyn_bonus, failed_pipes=None, current_round=0):
#         failed_pipes = failed_pipes or {}
#         LOOP_BALANCE_PIPE_TENURE = min(25, max(3, self.ctx.num_pipes // 5))
        
#         _, _, _, crit_node = self.ctx.get_cached_stats(indices)
#         if not crit_node or crit_node == "ERR": return None, None, "", -1

#         try: cycles = nx.cycle_basis(self.ctx.base_G_flow)
#         except: return None, None, "", -1
#         if not cycles: return None, None, "No cycles found", -1

#         progress = getattr(self.ctx, 'progress_ratio', 0.0)
        
#         if progress > 0.75:
#             max_allowed_drop = 1
#         elif progress > 0.40:
#             max_allowed_drop = 2
#         else:
#             max_allowed_drop = 3

#         best_drop_achieved = -1
#         candidates = []

#         import random
#         random.shuffle(cycles)

#         for cycle_nodes in cycles:
#             cycle_indices = []
#             full_cycle = cycle_nodes + [cycle_nodes[0]]
#             for u, v in zip(full_cycle[:-1], full_cycle[1:]):
#                 if (u, v) in self.ctx.edge_to_pipe: 
#                     cycle_indices.append(self.ctx.edge_to_pipe[(u, v)])

#             random.shuffle(cycle_indices)

#             for candidate_idx in cycle_indices:
#                 if (current_round - failed_pipes.get(candidate_idx, -999)) < LOOP_BALANCE_PIPE_TENURE:
#                     continue
                    
#                 curr_d_idx = indices[candidate_idx]
#                 if curr_d_idx < 2: continue 
                
#                 max_drop = min(max_allowed_drop, curr_d_idx) 
                
#                 for drop in range(max_drop, 0, -1):
#                     if drop < best_drop_achieved: continue
                        
#                     kicked, locked = list(indices), set()
#                     kicked[candidate_idx] -= drop
#                     locked.add(candidate_idx)
                    
#                     healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
#                     if is_feasible:
#                         test_squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
#                         sq_cost, _, _, _ = self.ctx.get_cached_stats(test_squeezed)
                        
#                         msg = f"FLOW STEER: Cut Pipe {candidate_idx + 1} (-{drop}). Healed {boosts}x."
#                         candidates.append((sq_cost, healed_sol, locked, msg, candidate_idx))
#                         best_drop_achieved = max(best_drop_achieved, drop)
            
#             if len(candidates) >= 5:
#                 break

#         if not candidates: return None, None, "FLOW STEER: Exhaustive search found no valid bypass.", -1
        
#         candidates.sort(key=lambda x: x[0])
#         chosen = random.choice(candidates[:3])
        
#         return chosen[1], chosen[2], chosen[3], chosen[4]

#     def ils_perturbation_kick(self, indices, stagnation_counter):        
#         pct = min(0.35, 0.08 + stagnation_counter * 0.01)
        
#         min_perturb = 2 if self.ctx.num_pipes < 100 else 4
#         n_perturb = max(min_perturb, int(self.ctx.num_pipes * pct))
        
#         is_late = stagnation_counter > 8
#         base_limit = 12 if self.ctx.num_pipes >= 200 else 40
        
#         if stagnation_counter > 15:
#             hard_limit = int(base_limit * 2.5)
#         else:
#             hard_limit = 25 if is_late else base_limit
            
#         n_perturb = min(n_perturb, hard_limit)
        
#         if self.ctx.num_pipes < 200:
#             chosen = random.sample(range(self.ctx.num_pipes), n_perturb)
#         else:
#             unit_losses = self.ctx.get_cached_heuristics(indices)
#             pipe_slack = []
#             for i in range(self.ctx.num_pipes):
#                 can_downgrade = indices[i] > 0
#                 slack_score = (1.0 / (unit_losses[i] + 1e-6)) if can_downgrade else 0.0
#                 pipe_slack.append((i, slack_score))
                
#             pipe_slack.sort(key=lambda x: x[1], reverse=True)
#             n_slack = int(self.ctx.num_pipes * 0.3) 
#             slack_pool = [p for p, _ in pipe_slack[:n_slack]]
#             tight_pool = [p for p, _ in pipe_slack[n_slack:]]
            
#             n_from_slack = max(1, int(n_perturb * 0.7))
#             n_from_tight = n_perturb - n_from_slack
            
#             chosen = random.sample(slack_pool, min(n_from_slack, len(slack_pool)))
#             if n_from_tight > 0 and tight_pool:
#                 chosen += random.sample(tight_pool, min(n_from_tight, len(tight_pool)))
            
#         kicked = list(indices)
#         locked = set()
        
#         for p_idx in chosen:
#             delta = random.choice([-1, 1]) 
#             new_val = max(0, min(self.ctx.max_d_idx, kicked[p_idx] + delta))
#             kicked[p_idx] = new_val
#             if delta > 0:
#                 locked.add(p_idx)
                
#         healed, ok, boosts = self.ls.heal_network(kicked, locked)
#         if not ok: return None, None, ""
        
#         msg = f"ILS-PERTURB (Slack-Aware): Shifted {len(chosen)} pipes. Healed {boosts}x."
#         return healed, locked, msg

#     def zero_sum_shift_kick(self, indices):
#         _, _, _, crit_node = self.ctx.get_cached_stats(indices)
#         if not crit_node or crit_node == "ERR": return None, None, ""

#         unit_losses = self.ctx.get_cached_heuristics(indices)
        
#         if self.ctx.num_pipes >= 200:
#             max_pairs = 1
#         else:
#             max_pairs = min(3, max(1, self.ctx.num_pipes // 20))
            
#         kicked = list(indices)
#         locked = set()
        
#         upgrade_candidates = []
#         for i in range(self.ctx.num_pipes):
#             if kicked[i] < self.ctx.max_d_idx:
#                 cost_diff = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]+1] - self.ctx.costs_array[kicked[i]])
#                 score = unit_losses[i] / max(cost_diff, 1.0)
#                 upgrade_candidates.append((i, cost_diff, score))
                
#         downgrade_candidates = []
#         for i in range(self.ctx.num_pipes):
#             if kicked[i] > 0:
#                 cost_diff = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]] - self.ctx.costs_array[kicked[i]-1])
#                 score = cost_diff / max(unit_losses[i], 1e-5)
#                 downgrade_candidates.append((i, cost_diff, score))
                
#         upgrade_candidates.sort(key=lambda x: x[2], reverse=True)
#         downgrade_candidates.sort(key=lambda x: x[2], reverse=True)
        
#         exchanges = 0
#         used_pipes = set()
        
#         for up_pipe, up_cost, _ in upgrade_candidates:
#             if exchanges >= max_pairs: break
#             if up_pipe in used_pipes: continue
            
#             best_down_pipe = -1
#             best_diff = float('inf')
            
#             for down_pipe, down_cost, _ in downgrade_candidates:
#                 if down_pipe in used_pipes or down_pipe == up_pipe: continue
                
#                 cost_balance = abs(up_cost - down_cost)
#                 if cost_balance < best_diff and cost_balance < (up_cost * 0.5):
#                     best_diff = cost_balance
#                     best_down_pipe = down_pipe
                    
#             if best_down_pipe != -1:
#                 kicked[up_pipe] += 1
#                 kicked[best_down_pipe] -= 1
#                 locked.add(up_pipe)
#                 locked.add(best_down_pipe)
#                 used_pipes.add(up_pipe)
#                 used_pipes.add(best_down_pipe)
#                 exchanges += 1

#         if not locked: return None, None, ""
#         return kicked, locked, f"ZERO-SUM: Exchanged {exchanges} pairs (Cost-Optimized)."
    
#     def submarine_oscillation_kick(self, indices):
#         n = self.ctx.num_pipes
#         unit_losses = self.ctx.get_cached_heuristics(indices)
#         mid_d = self.ctx.max_d_idx // 2
        
#         mains = [i for i in range(n) if indices[i] <= mid_d and unit_losses[i] < 0.05]
#         if not mains:
#             mains = [i for i in range(n) if indices[i] <= mid_d]
            
#         peripherals = [i for i in range(n) if indices[i] >= (mid_d + 1) and unit_losses[i] > 0.05]
#         if not peripherals:
#             peripherals = [i for i in range(n) if indices[i] >= (mid_d + 1)]
            
#         if not mains or not peripherals:
#             return None, None, "No valid Submarine targets found"
            
#         import random
#         kicked = list(indices)
        
#         progress = getattr(self.ctx, 'progress_ratio', 0.0)
#         shift = 1
        
#         # 1. Зрізаємо магістралі
#         n_mains = 1 if n < 100 else min(2, len(mains))
#         target_mains = random.sample(mains, n_mains)
#         for p in target_mains:
#             kicked[p] = min(self.ctx.max_d_idx, kicked[p] + shift)
            
#         # 2. Розширюємо периферію
#         n_periph = min(3, len(peripherals))
#         target_periph = random.sample(peripherals, n_periph)
#         for p in target_periph:
#             kicked[p] = max(0, kicked[p] - shift)
            
#         healed, ok, boosts = self.ls.heal_network(kicked, set())
#         if not ok: return None, None, f"Submarine oscillation unhealable (tried shift ±{shift})"
        
#         return healed, set(), f"SUBMARINE: Cut {n_mains} mains, Boosted {n_periph} periphs. Healed {boosts}x."
    
#     def basin_escape(self, indices, global_archive):
#         if not global_archive or len(global_archive) < 2:
#             return None, None, "Archive too small"

#         best_dist = -1
#         diverse_sol = None
#         for _, arch_sol in global_archive:
#             dist = sum(1 for a, b in zip(indices, arch_sol) if a != b)
#             if dist > best_dist:
#                 best_dist = dist
#                 diverse_sol = arch_sol

#         min_dist = max(2, int(self.ctx.num_pipes * 0.03))
        
#         if diverse_sol is None or best_dist < min_dist:
#             return None, None, "No diverse target"

#         diff_pipes = [i for i in range(self.ctx.num_pipes) if indices[i] != diverse_sol[i]]
#         downgrade_pipes = [i for i in diff_pipes if diverse_sol[i] < indices[i]]
#         target_pipes = downgrade_pipes if len(downgrade_pipes) >= 5 else diff_pipes
        
#         min_rep = 2 if self.ctx.num_pipes < 100 else 5
#         n_replace = max(min_rep, min(len(target_pipes) // 4, 20))
        
#         import random
#         replace_pipes = random.sample(target_pipes, n_replace)
        
#         kicked = list(indices)
#         for p in replace_pipes:
#             kicked[p] = diverse_sol[p]

#         healed, ok, boosts = self.ls.heal_network(kicked, set())
#         if not ok:
#             return None, None, "Heal failed"

#         return healed, set(), f"BASIN-ESCAPE: Transplanted {n_replace} cluster pipes. Healed {boosts}x."