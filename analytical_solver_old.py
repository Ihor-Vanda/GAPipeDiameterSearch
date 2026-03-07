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
        
        # ОПТИМІЗАЦІЯ 1: Кешування індексів діаметрів (O(1) замість O(N))
        self._diam_to_idx = {d: i for i, d in enumerate(self.diameters)}
        
        # ОПТИМІЗАЦІЯ 2: Попередньо побудований граф для маршрутизації
        self._base_G_flow = nx.Graph()
        self._edge_to_pipe = {}
        for u, v, k in self.simulator.graph.edges(keys=True):
            if k in self.simulator.component_names:
                idx = self.simulator.component_names.index(k)
                self._base_G_flow.add_edge(u, v)
                self._edge_to_pipe[(u, v)] = idx
                self._edge_to_pipe[(v, u)] = idx
                
        # ОПТИМІЗАЦІЯ 3: Кеш результатів EPANET
        self._sim_cache = {}
        self._heuristic_cache = {} # Також кешуємо евристики тертя
        
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
        cost, _, _, _ = self._get_cached_stats(idx)
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
        # O(1) доступ через словник
        return [self._diam_to_idx.get(d, len(self.diameters)-1) for d in solution]
    
    def _get_cached_stats(self, indices):
        """Обгортка над EPANET з кешуванням результатів."""
        sig = tuple(indices)
        if sig not in self._sim_cache:
            # ТУТ МАЄ БУТИ ВИКЛИК СИМУЛЯТОРА, А НЕ СЕБЕ:
            self._sim_cache[sig] = self.simulator.get_stats(indices)
        return self._sim_cache[sig]
        
    def _get_cached_heuristics(self, indices):
        """Кешована обгортка для евристик тертя."""
        sig = tuple(indices)
        if sig not in self._heuristic_cache:
            # ТУТ ТАКОЖ ВИКЛИК СИМУЛЯТОРА:
            self._heuristic_cache[sig] = self.simulator.get_heuristics(indices)
        return self._heuristic_cache[sig]

    def _gradient_squeeze(self, solution, locked_pipes=None, max_passes=None, quick_mode=False, verbose=False):
        """
        COMPENSATORY SQUEEZE:
        Attempts to greedily reduce pipe sizes. If a reduction fails due to pressure,
        it tries to 'buy' validity by expanding a cheaper downstream neighbor.
        """
        locked_pipes = locked_pipes or set()
        current_solution = list(solution)
        improved = True
        
        lengths = self.simulator.lengths
        costs_array = self.simulator.costs 
        passes = 0
        active_indices = [i for i in range(self.num_pipes) if i not in locked_pipes]
        
        curr_indices = self._get_current_indices(current_solution)
        cost, p_min, _, _ = self._get_cached_stats(curr_indices)
        
        if verbose:
            surplus = p_min - self.simulator.config.h_min
            self._log(f"   [SQUEEZE] Start: {cost/1e6:.4f}M. Headroom: {surplus:.2f}m")

        while improved:
            improved = False
            passes += 1
            if max_passes and passes > max_passes: break
            
            curr_indices = self._get_current_indices(current_solution)
            unit_losses = self._get_cached_heuristics(curr_indices)
            savings_candidates = []
            
            for i in active_indices:
                if quick_mode and unit_losses[i] > 0.05: continue
                
                current_idx = curr_indices[i]
                if current_idx > 0:
                    dollar_save = lengths[i] * (costs_array[current_idx] - costs_array[current_idx - 1])
                    if dollar_save < 50.0: continue
                    
                    risk = unit_losses[i] + 1e-9
                    score = dollar_save / risk
                    savings_candidates.append((i, score, dollar_save))
            
            if not savings_candidates: break

            savings_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates_to_try = savings_candidates[:5 if quick_mode else 20]

            batch_count = 0
            
            for idx, _, save in candidates_to_try:
                current_idx = self.diameters.index(current_solution[idx])
                test_solution = list(current_solution)
                test_solution[idx] = self.diameters[current_idx - 1]
                
                t_indices = self._get_current_indices(test_solution)
                _, t_p, t_feas, t_crit = self._get_cached_stats(t_indices)
                
                if t_feas and t_p >= self.simulator.config.h_min:
                    current_solution = test_solution
                    p_min = t_p
                    improved = True
                    batch_count += 1
                    if batch_count >= 5: break
                
                # Compensatory Logic (Expand neighbor if reduction failed)
                elif not quick_mode and save > (self.adaptive_bonus * 0.5) and t_crit is not None:
                    neighbors = []
                    for u, v, k in self.simulator.graph.edges(keys=True):
                        if (u == t_crit or v == t_crit) and k in self.simulator.component_names:
                            n_idx = self.simulator.component_names.index(k)
                            if n_idx != idx and n_idx not in locked_pipes:
                                neighbors.append(n_idx)
                    
                    best_fix = None
                    best_net_save = 0
                    best_neighbor = -1
                    best_p = p_min
                    
                    for n_idx in neighbors:
                        n_curr_d = self.diameters.index(test_solution[n_idx])
                        if n_curr_d < len(self.diameters) - 1:
                            fix_solution = list(test_solution)
                            fix_solution[n_idx] = self.diameters[n_curr_d + 1] 
                            
                            expansion_cost = lengths[n_idx] * (costs_array[n_curr_d + 1] - costs_array[n_curr_d])
                            net_save = save - expansion_cost
                            
                            if net_save > (self.adaptive_bonus * 0.5): 
                                f_indices = self._get_current_indices(fix_solution)
                                _, f_p, f_feas, _ = self._get_cached_stats(f_indices)
                                
                                if f_feas and f_p >= self.simulator.config.h_min:
                                    if net_save > best_net_save:
                                        best_net_save = net_save
                                        best_fix = fix_solution
                                        best_neighbor = n_idx
                                        best_p = f_p

                    if best_fix:
                        current_solution = best_fix
                        p_min = best_p
                        improved = True
                        batch_count += 1
                        if verbose:
                            self._log(f"      -> [COMBO] Reduced {idx+1} & Expanded {best_neighbor+1}. Net Save: ${best_net_save:,.0f}")
                        if batch_count >= 5: break

        return current_solution
    
    def _get_lazy_periphery(self, solution):
        """
        Helper method: Finds the dominant path, protects mainlines (top 40%),
        and returns a sorted list of the most 'lazy' periphery pipes.
        """
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self._get_cached_stats(curr_indices)
        if not crit_node or crit_node == "ERR": 
            return curr_indices, []

        path_pipes, _ = self._get_dominant_path(curr_indices, crit_node)
        unit_losses = self._get_cached_heuristics(curr_indices)
        
        # УНІВЕРСАЛЬНІСТЬ: Периферією вважаються труби з нижніх 60% каталогу
        threshold_idx = int(len(self.diameters) * 0.6)
        
        periphery_pipes = [i for i in range(self.num_pipes) 
                           if i not in path_pipes and curr_indices[i] <= threshold_idx]
        periphery_pipes.sort(key=lambda i: unit_losses[i])
        
        return curr_indices, periphery_pipes
    
    def _micro_trim_kick(self, solution):
        curr_indices, periphery_pipes = self._get_lazy_periphery(solution)
        if not periphery_pipes: return None, None, ""
        
        kicked = list(solution)
        locked = set()
        shrunk_count = 0
        
        for idx in periphery_pipes:
            curr_d_idx = curr_indices[idx]
            if curr_d_idx > 0: 
                kicked[idx] = self.diameters[curr_d_idx - 1]
                locked.add(idx)
                shrunk_count += 1
                
            if shrunk_count >= 5: 
                break
                
        if shrunk_count > 0:
            healed_sol, is_feasible, boosts = self._heal_network(kicked, locked)
            if is_feasible:
                return healed_sol, locked, f"FINISHER: Force-shrunk {shrunk_count} periphery pipes. Healed {boosts}x."
            
        return None, None, ""
    
    # def _sync_trim_kick(self, solution):
    #     curr_indices, periphery_pipes = self._get_lazy_periphery(solution)
    #     if not periphery_pipes: return None, None, ""
        
    #     best_sol = None
    #     best_cost = float('inf')
    #     locked_best = set()
    #     boosts_best = 0
        
    #     for d1, d2 in itertools.combinations(periphery_pipes[:15], 2):
    #         d1_idx, d2_idx = curr_indices[d1], curr_indices[d2]
    #         if d1_idx == 0 or d2_idx == 0: continue
            
    #         test_sol = list(solution)
    #         test_sol[d1] = self.diameters[d1_idx - 1]
    #         test_sol[d2] = self.diameters[d2_idx - 1]
    #         locked = {d1, d2}
            
    #         healed_sol, is_feasible, boosts = self._heal_network(test_sol, locked)
            
    #         if is_feasible:
    #             squeezed = self._gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True)
    #             sq_idx = self._get_current_indices(squeezed)
    #             sq_c, _, _, _ = self._get_cached_stats(sq_idx)
                
    #             if sq_c < best_cost:
    #                 best_cost, best_sol, locked_best, boosts_best = sq_c, healed_sol, locked, boosts
                    
    #     if best_sol:
    #         return best_sol, locked_best, f"SYNC-TRIM: Shrunk Periphery Pipes {[p+1 for p in locked_best]}. Healed {boosts_best}x."
            
    #     return None, None, ""
    
    def _sync_trim_kick(self, solution):
        import random
        
        curr_indices, periphery_pipes = self._get_lazy_periphery(solution)
        if not periphery_pipes: return None, None, ""
        
        candidates = []
        
        # 1. СУРРОГАТНИЙ ФІЛЬТР (Швидко перебираємо всі комбінації)
        for d1, d2 in itertools.combinations(periphery_pipes[:15], 2):
            d1_idx = curr_indices[d1]
            d2_idx = curr_indices[d2]
            if d1_idx == 0 or d2_idx == 0: continue
            
            test_sol = list(solution)
            test_sol[d1] = self.diameters[d1_idx - 1]
            test_sol[d2] = self.diameters[d2_idx - 1]
            locked = {d1, d2}
            
            healed_sol, is_feasible, boosts = self._heal_network(test_sol, locked)
            
            if is_feasible:
                h_idx = self._get_current_indices(healed_sol)
                h_c, h_p, _, _ = self._get_cached_stats(h_idx) 
                
                # Швидкий математичний прогноз
                h_surplus = h_p - self.simulator.config.h_min
                h_score = h_c - (h_surplus * self.adaptive_bonus)
                
                candidates.append((h_score, healed_sol, locked, boosts))
                
        if not candidates: return None, None, ""
        
        # Сортуємо всіх кандидатів за нашим прогнозом (від найперспективніших)
        candidates.sort(key=lambda x: x[0])
        
        validated_candidates = []
        
        # 2. ФІЗИЧНА ВАЛІДАЦІЯ (Пропускаємо через Squeeze лише ТОП-5 прогнозів)
        for _, healed_sol, locked, boosts in candidates[:5]:
            # Симулятор покаже реальну, а не спрогнозовану ціну
            squeezed = self._gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True)
            sq_idx = self._get_current_indices(squeezed)
            sq_c, _, _, _ = self._get_cached_stats(sq_idx)
            
            # Збираємо всіх валідованих кандидатів з їхньою реальною ціною
            validated_candidates.append((sq_c, healed_sol, locked, boosts))
                
        if not validated_candidates:
            return None, None, ""
            
        # 3. РАНДОМІЗАЦІЯ ДЛЯ ЗАХИСТУ ВІД ПЕТЕЛЬ
        # Сортуємо за РЕАЛЬНОЮ ціною після сквізу
        validated_candidates.sort(key=lambda x: x[0])
        
        # Беремо ТОП-3 найкращих реальних результатів (або менше, якщо є лише 1 чи 2)
        top_validated = validated_candidates[:3]
        
        # ВИПАДКОВО обираємо одного з Топ-3
        chosen = random.choice(top_validated)
        _, best_final_sol, best_locked, best_boosts = chosen
        
        return best_final_sol, best_locked, f"SYNC-TRIM: Shrunk Pipes {[p+1 for p in best_locked]}. Healed {best_boosts}x."
    
    # def _sync_trim_kick(self, solution):
    #     curr_indices, periphery_pipes = self._get_lazy_periphery(solution)
    #     if not periphery_pipes: return None, None, ""
        
    #     candidates = []
        
    #     # 1. СУРРОГАТНИЙ ФІЛЬТР (Швидко перебираємо всі комбінації)
    #     for d1, d2 in itertools.combinations(periphery_pipes[:15], 2):
    #         d1_idx = curr_indices[d1]
    #         d2_idx = curr_indices[d2]
    #         if d1_idx == 0 or d2_idx == 0: continue
            
    #         test_sol = list(solution)
    #         test_sol[d1] = self.diameters[d1_idx - 1]
    #         test_sol[d2] = self.diameters[d2_idx - 1]
    #         locked = {d1, d2}
            
    #         healed_sol, is_feasible, boosts = self._heal_network(test_sol, locked)
            
    #         if is_feasible:
    #             h_idx = self._get_current_indices(healed_sol)
    #             h_c, h_p, _, _ = self._get_cached_stats(h_idx) 
                
    #             # Швидкий математичний прогноз
    #             h_surplus = h_p - self.simulator.config.h_min
    #             h_score = h_c - (h_surplus * self.adaptive_bonus)
                
    #             candidates.append((h_score, healed_sol, locked, boosts))
                
    #     if not candidates: return None, None, ""
        
    #     # Сортуємо всіх кандидатів за нашим прогнозом (від найперспективніших)
    #     candidates.sort(key=lambda x: x[0])
        
    #     best_sq_cost = float('inf')
    #     best_final_sol = None
    #     best_locked = set()
    #     best_boosts = 0
        
    #     # 2. ФІЗИЧНА ВАЛІДАЦІЯ (Пропускаємо через Squeeze лише ТОП-5)
    #     for _, healed_sol, locked, boosts in candidates[:5]:
    #         # Симулятор покаже реальну, а не спрогнозовану ціну
    #         squeezed = self._gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True)
    #         sq_idx = self._get_current_indices(squeezed)
    #         sq_c, _, _, _ = self._get_cached_stats(sq_idx)
            
    #         if sq_c < best_sq_cost:
    #             best_sq_cost = sq_c
    #             # ПРИМІТКА: Повертаємо healed_sol (як у старому коді), 
    #             # оскільки головний цикл все одно застосує до нього глибокий 8-pass Squeeze
    #             best_final_sol = healed_sol 
    #             best_locked = locked
    #             best_boosts = boosts
                
    #     if best_final_sol:
    #         return best_final_sol, best_locked, f"SYNC-TRIM: Shrunk Pipes {[p+1 for p in best_locked]}. Healed {best_boosts}x."
            
    #     return None, None, ""
    
    def _make_diverse_seeds(self):
        """
        Генерує різноманітні стартові точки для Beam Search.
        ГАРАНТУЄ, що кожне насіння є гідравлічно життєздатним (або відсіює його).
        """
        seeds = []
        velocities_to_try = [1.0, 1.2, 0.8] # Стандарт, Агресивний (тонкі труби), Консервативний
        
        for v in velocities_to_try:
            self.v_opt = v
            sol = [self.diameters[-1]] * self.num_pipes
            for _ in range(self.max_iters):
                flows, _ = self.simulator.get_hydraulic_state(sol)
                new = [self._calculate_ideal_d(q) for q in flows]
                if new == sol: break
                sol = new
                
            # 1. ПЕРЕВІРКА ТА ЛІКУВАННЯ
            idx = self._get_current_indices(sol)
            _, p, feas, _ = self._get_cached_stats(idx)
            
            if not feas or p < self.simulator.config.h_min:
                sol, is_healed, _ = self._heal_network(sol, set())
                # ВІДСІЮВАННЯ: якщо навіть Лікар не врятував мережу, пропускаємо це насіння
                if not is_healed: 
                    self._log(f"      [SEED] Dropped invalid seed for v_opt={v}")
                    continue 
                    
            # 2. Швидке стискання життєздатної мережі
            squeezed_sol = self._gradient_squeeze(sol, quick_mode=True)
            seeds.append(squeezed_sol)
            
        self.v_opt = 1.0 # Повертаємо стандартну швидкість
        
        # # --- Топологічне насіння (Магістралі = Max, Периферія = Min) ---
        # topo_sol = [self.diameters[-1]] * self.num_pipes
        # curr_idx = self._get_current_indices(topo_sol)
        # _, _, _, crit_node = self._get_cached_stats(curr_idx)
        
        # if crit_node and crit_node != "ERR":
        #     path_pipes, _ = self._get_dominant_path(curr_idx, crit_node)
        #     for i in range(self.num_pipes):
        #         if i not in path_pipes:
        #             topo_sol[i] = self.diameters[1] # Беремо другий найменший діаметр
                    
        #     # 1. ПЕРЕВІРКА ТА ЛІКУВАННЯ ТОПОЛОГІЧНОГО НАСІННЯ
        #     idx = self._get_current_indices(topo_sol)
        #     _, p, feas, _ = self._get_cached_stats(idx)
            
        #     if not feas or p < self.simulator.config.h_min:
        #         topo_sol, is_healed, _ = self._heal_network(topo_sol, set())
        #         if is_healed:
        #             seeds.append(self._gradient_squeeze(topo_sol, quick_mode=True))
        #         else:
        #             self._log("      [SEED] Dropped invalid topological seed.")
        #     else:
        #         seeds.append(self._gradient_squeeze(topo_sol, quick_mode=True))

        return seeds
    
    # def _make_diverse_seeds(self):
    #     seeds = []
    #     sol1 = [self.diameters[-1]] * self.num_pipes
    #     for _ in range(self.max_iters):
    #         flows, _ = self.simulator.get_hydraulic_state(sol1)
    #         new = [self._calculate_ideal_d(q) for q in flows]
    #         if new == sol1: break
    #         sol1 = new
    #     seeds.append(self._gradient_squeeze(sol1))
        
    #     sol2 = [self.diameters[-1]] * self.num_pipes
    #     seeds.append(self._gradient_squeeze(sol2, quick_mode=True))
        
    #     return seeds

    def _forcing_hand_kick(self, solution):
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self._get_cached_stats(curr_indices)
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
        _, _, _, crit_node = self._get_cached_stats(curr_indices)
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
            unit_losses = self._get_cached_heuristics(curr_indices)
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

        if target_idx == curr_d_idx: return None, None, "" # Запобіжник від холостих ходів

        kicked[best_candidate] = self.diameters[target_idx]
        locked.add(best_candidate)
        
        return kicked, locked, f"BOTTLENECK: Boosted Pipe {best_candidate} ({reason}) to {self.diameters[target_idx]}"

    def _topological_inversion_kick(self, solution, tabu_set=None):
        """
        TOPOLOGICAL DIVERSITY SEARCH (Relaxed):
        If strict alternatives are not found, allows paths with higher overlap.
        """
        if tabu_set is None: tabu_set = set()
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self._get_cached_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return None, None, "", None
        
        # 1. Викликаємо наш новий універсальний метод! Він сам знайде правильне джерело.
        dominant_indices, dominant_path_nodes = self._get_dominant_path(curr_indices, crit_node)
        if not dominant_path_nodes: return None, None, "", None
        
        # 2. ТЕПЕР це абсолютно безпечно: беремо резервуар з початку знайденого шляху
        source = dominant_path_nodes[0]
        
        # 3. Збираємо дані для аналізу топології
        dominant_edges = set()
        dom_diameters = []
        for i, (u, v) in enumerate(zip(dominant_path_nodes[:-1], dominant_path_nodes[1:])):
            idx = dominant_indices[i]
            dominant_edges.add(tuple(sorted((u, v))))
            dom_diameters.append(curr_indices[idx])
            
        target_capacity_idx = int(np.mean(dom_diameters)) if dom_diameters else len(self.diameters)-1
        
        # 4. Будуємо граф для пошуку альтернативних шляхів (обхідних кілець)
        G_simple = nx.Graph()
        for u, v in self.simulator.graph.edges(): G_simple.add_edge(u, v)
        
        candidates = []
        try:
            path_generator = nx.shortest_simple_paths(G_simple, source, crit_node)
            candidates = list(itertools.islice(path_generator, 100)) # Більше кандидатів
        except: return None, None, "", None

        scored_paths = []
        
        # ... (ДАЛІ ВАШ КОД ЗАЛИШАЄТЬСЯ БЕЗ ЗМІН, починаючи з: for p_nodes in candidates:) ...
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
        """
        SMART COMBO SWAP (Predictive Scoring):
        Upgrades pipes ON the critical path to buy pressure.
        Downgrades ANY lazy pipe in the network to save money.
        Uses cached fast-evaluation to score all combinations, 
        and only applies the heavy Squeeze to the absolute best candidate.
        """
        curr_indices = self._get_current_indices(solution)
        cost, p_min, _, crit_node = self._get_cached_stats(curr_indices)
        best_sol = list(solution)
        best_found_cost = cost
        
        # Якщо немає критичного вузла, повертаємо базовий оптимум
        if not crit_node or crit_node == "ERR": return best_sol

        # 1. Знаходимо шлях до проблеми (Кандидати на РОЗШИРЕННЯ)
        path_pipes, _ = self._get_dominant_path(curr_indices, crit_node)
        if not path_pipes: return best_sol
        
        # 2. Знаходимо "ледачі" труби (Кандидати на ЗВУЖЕННЯ)
        unit_losses = self._get_cached_heuristics(curr_indices)
        lazy_pipes = sorted(range(self.num_pipes), key=lambda i: unit_losses[i])
        
        candidates = []
        
        # --- ТИП 1: 1-UP (Магістраль), 1-DOWN (Будь-який ледар) ---
        for up_pipe in path_pipes[-15:]:
            u_idx = self.diameters.index(best_sol[up_pipe])
            if u_idx == len(self.diameters) - 1: continue
            
            for down_pipe in lazy_pipes[:20]:
                if up_pipe == down_pipe: continue
                
                d_idx = self.diameters.index(best_sol[down_pipe])
                if d_idx == 0: continue
                
                test_sol = list(best_sol)
                test_sol[up_pipe] = self.diameters[u_idx + 1]
                test_sol[down_pipe] = self.diameters[d_idx - 1]
                
                # Швидка перевірка через кеш (БЕЗ Squeeze)
                t_idx = self._get_current_indices(test_sol)
                c, p, feas, _ = self._get_cached_stats(t_idx)
                
                if feas and p >= self.simulator.config.h_min:
                    surplus = p - self.simulator.config.h_min
                    score = c - (surplus * self.adaptive_bonus)
                    candidates.append((score, test_sol, {up_pipe}))

        # --- ТИП 2: COMBO 1-UP (Магістраль), 2-DOWN (Будь-які ледарі) ---
        for up_pipe in path_pipes[-10:]:
            u_idx = self.diameters.index(best_sol[up_pipe])
            if u_idx == len(self.diameters) - 1: continue
            
            for d1, d2 in itertools.combinations(lazy_pipes[:15], 2):
                if up_pipe in (d1, d2): continue
                
                d1_idx = self.diameters.index(best_sol[d1])
                d2_idx = self.diameters.index(best_sol[d2])
                if d1_idx == 0 or d2_idx == 0: continue
                
                test_sol = list(best_sol)
                test_sol[up_pipe] = self.diameters[u_idx + 1]
                test_sol[d1] = self.diameters[d1_idx - 1]
                test_sol[d2] = self.diameters[d2_idx - 1]
                
                # Швидка перевірка через кеш
                t_idx = self._get_current_indices(test_sol)
                c, p, feas, _ = self._get_cached_stats(t_idx)
                
                if feas and p >= self.simulator.config.h_min:
                    surplus = p - self.simulator.config.h_min
                    score = c - (surplus * self.adaptive_bonus)
                    candidates.append((score, test_sol, {up_pipe}))

       # Якщо жодних валидних замін не знайдено
        if not candidates: 
            return best_sol
            
        # Сортуємо кандидатів за нашим прогнозом (Score)
        candidates.sort(key=lambda x: x[0])
        
        # --- ПАСТКА 1 ВИРІШЕНА: Рандом з Топ-3 замість абсолютної жадібності ---
        import random
        top_candidates = candidates[:3]
        chosen = random.choice(top_candidates)
        best_score, best_candidate_sol, locked = chosen
        
        # Шліфуємо тільки обраного
        squeezed = self._gradient_squeeze(best_candidate_sol, locked_pipes=locked, max_passes=2, quick_mode=True)
        sq_idx = self._get_current_indices(squeezed)
        sq_c, _, _, _ = self._get_cached_stats(sq_idx)
        
        # Якщо після Сквізу рішення дійсно дешевше за поточний рекорд - повертаємо його
        if sq_c < best_found_cost:
            return squeezed
            
        return best_sol
    
    def _get_dominant_path(self, curr_indices, crit_node):
        try:
            # Просто оновлюємо ваги у вже існуючому графі (дуже швидко)
            for u, v in self._base_G_flow.edges():
                idx = self._edge_to_pipe[(u, v)]
                self._base_G_flow[u][v]['weight'] = 100.0 / (curr_indices[idx] + 1)
                    
            best_path_nodes = []
            best_weight = float('inf')
            
            for source in self.simulator.sources:
                try:
                    path = nx.shortest_path(self._base_G_flow, source, crit_node, weight='weight')
                    # Рахуємо вагу знайденого шляху
                    path_weight = sum(self._base_G_flow[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    if path_weight < best_weight:
                        best_weight = path_weight
                        best_path_nodes = path
                except nx.NetworkXNoPath:
                    continue 
            
            if not best_path_nodes: return [], []
            
            path_pipes = [self._edge_to_pipe[(u, v)] for u, v in zip(best_path_nodes[:-1], best_path_nodes[1:])]
            return path_pipes, best_path_nodes
            
        except Exception as e:
            return [], []
        
    def _heal_network(self, solution, locked_pipes):
        """
        UNIVERSAL HEALER:
        If a network is infeasible (P < 30m), it finds the failing node, 
        traces the path back to the source, and widens the worst bottleneck 
        on that path. Repeats until the network survives.
        """
        kicked = list(solution)
        boosts = 0
        for _ in range(40): # Максимум 40 рятувальних розширень
            test_indices = self._get_current_indices(kicked)
            _, t_p, t_feas, crit_node = self._get_cached_stats(test_indices)
            
            # Якщо мережа вилікувана - повертаємо успіх
            if t_feas and t_p >= self.simulator.config.h_min:
                return kicked, True, boosts
            
            # Знаходимо шлях до вузла, який зараз "задихається"
            path_pipes, _ = self._get_dominant_path(test_indices, crit_node)
            if not path_pipes: return kicked, False, boosts
            
            unit_losses = self._get_cached_heuristics(test_indices)
            worst_pipe = -1
            max_loss = -1.0
            
            # Шукаємо найвужче місце на цьому шляху
            for idx in path_pipes:
                if idx in locked_pipes: continue # Не чіпаємо те, що щойно спеціально відрізали
                if test_indices[idx] < len(self.diameters) - 1: # Якщо ще можна розширити
                    if unit_losses[idx] > max_loss:
                        max_loss = unit_losses[idx]
                        worst_pipe = idx
            
            # Розширюємо найгіршу трубу на 1 розмір і пробуємо знову
            if worst_pipe != -1:
                kicked[worst_pipe] = self.diameters[test_indices[worst_pipe] + 1]
                boosts += 1
            else:
                # Всі труби на шляху вже 40", лікувати нічим
                break
        
        return kicked, False, boosts
    
    def _loop_balancing_kick(self, solution):
        import random # Не забудьте імпорт!
        
        curr_indices = self._get_current_indices(solution)
        _, _, _, crit_node = self._get_cached_stats(curr_indices)
        if not crit_node or crit_node == "ERR": return None, None, ""

        G_simple = nx.Graph()
        edge_to_pipe = {}
        for u, v, k in self.simulator.graph.edges(keys=True):
            if k in self.simulator.component_names:
                idx = self.simulator.component_names.index(k)
                G_simple.add_edge(u, v)
                edge_to_pipe[(u, v)] = idx
                edge_to_pipe[(v, u)] = idx

        try: cycles = nx.cycle_basis(G_simple)
        except: return None, None, ""
        if not cycles: return None, None, "No cycles found"

        best_drop_achieved = -1
        
        # ЗБИРАЄМО ВСІХ КАНДИДАТІВ СЮДИ:
        candidates = []

        for cycle_nodes in cycles:
            cycle_indices = []
            full_cycle = cycle_nodes + [cycle_nodes[0]]
            for u, v in zip(full_cycle[:-1], full_cycle[1:]):
                if (u, v) in edge_to_pipe:
                    cycle_indices.append(edge_to_pipe[(u, v)])

            for candidate_idx in cycle_indices:
                curr_d_idx = curr_indices[candidate_idx]
                if curr_d_idx < 2: continue 
                
                max_drop = min(3, curr_d_idx) 
                
                for drop in range(max_drop, 0, -1):
                    if drop < best_drop_achieved:
                        continue
                        
                    target_d_idx = curr_d_idx - drop
                    kicked = list(solution)
                    locked = set()
                    
                    kicked[candidate_idx] = self.diameters[target_d_idx]
                    locked.add(candidate_idx)
                    
                    healed_sol, is_feasible, boosts = self._heal_network(kicked, locked)
                    
                    if is_feasible:
                        test_squeezed = self._gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=4, quick_mode=True)
                        sq_indices = self._get_current_indices(test_squeezed)
                        sq_cost, _, _, _ = self._get_cached_stats(sq_indices)
                        
                        msg = f"FLOW STEER: Cut Pipe {candidate_idx + 1} (-{drop}). Healed {boosts}x."
                        # ЗБЕРІГАЄМО КАНДИДАТА
                        candidates.append((sq_cost, healed_sol, locked, msg))

        if not candidates:
            return None, None, "FLOW STEER: Exhaustive search found no valid bypass."
            
        # 1. Сортуємо від найдешевшого до найдорожчого
        candidates.sort(key=lambda x: x[0])
        
        # 2. Беремо ТОП-3 (або менше, якщо стільки не знайшлося)
        top_candidates = candidates[:3]
        
        # 3. ВИПАДКОВО обираємо одного з Топ-3
        chosen = random.choice(top_candidates)
        
        return chosen[1], chosen[2], chosen[3]
    
    # def _loop_balancing_kick(self, solution):
    #     """
    #     FLOW STEERING (Exhaustive Cut & Heal):
    #     Finds all cycles. Tests cutting EVERY large pipe. 
    #     Uses the Universal Healer to fix the network. 
    #     Selects the cut that yields the absolute cheapest final network.
    #     """
    #     curr_indices = self._get_current_indices(solution)
    #     _, _, _, crit_node = self._get_cached_stats(curr_indices)
    #     if not crit_node or crit_node == "ERR": return None, None, ""

    #     G_simple = nx.Graph()
    #     edge_to_pipe = {}
    #     for u, v, k in self.simulator.graph.edges(keys=True):
    #         if k in self.simulator.component_names:
    #             idx = self.simulator.component_names.index(k)
    #             G_simple.add_edge(u, v)
    #             edge_to_pipe[(u, v)] = idx
    #             edge_to_pipe[(v, u)] = idx

    #     try: cycles = nx.cycle_basis(G_simple)
    #     except: return None, None, ""
    #     if not cycles: return None, None, "No cycles found"

    #     best_combo_sol = None
    #     best_combo_locked = set()
    #     best_combo_score = float('inf')
    #     best_msg = ""
    #     best_drop_achieved = -1

    #     for cycle_nodes in cycles:
    #         cycle_indices = []
    #         full_cycle = cycle_nodes + [cycle_nodes[0]]
    #         for u, v in zip(full_cycle[:-1], full_cycle[1:]):
    #             if (u, v) in edge_to_pipe:
    #                 cycle_indices.append(edge_to_pipe[(u, v)])

    #         # Пробуємо відрізати КОЖНУ велику трубу в кільці
    #         for candidate_idx in cycle_indices:
    #             curr_d_idx = curr_indices[candidate_idx]
    #             if curr_d_idx < 2: continue # Дрібні труби не чіпаємо
                
    #             max_drop = min(3, curr_d_idx) 
                
    #             # Починаємо з радикальних відрізань (-3, потім -2, потім -1)
    #             for drop in range(max_drop, 0, -1):
    #                 # Якщо ми вже знайшли глибокий поріз, не витрачаємо час на мілкі
    #                 if drop < best_drop_achieved:
    #                     continue
                        
    #                 target_d_idx = curr_d_idx - drop
    #                 kicked = list(solution)
    #                 locked = set()
                    
    #                 # 1. Робимо розріз
    #                 kicked[candidate_idx] = self.diameters[target_d_idx]
    #                 locked.add(candidate_idx)
                    
    #                 # 2. Кличемо Лікаря, щоб він врятував тиск
    #                 healed_sol, is_feasible, boosts = self._heal_network(kicked, locked)
                    
    #                 if is_feasible:
    #                     # 3. Швидко стискаємо "жир" (даємо 4 проходи для кращої очистки)
    #                     test_squeezed = self._gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=4, quick_mode=True)
    #                     sq_indices = self._get_current_indices(test_squeezed)
    #                     sq_cost, _, _, _ = self._get_cached_stats(sq_indices)
                        
    #                     # ВИПРАВЛЕННЯ: Шукаємо абсолютний мінімум ціни. Ніякого пріоритету на глибину зрізу!
    #                     if sq_cost < best_combo_score:
    #                         best_combo_score = sq_cost
    #                         best_combo_sol = healed_sol
    #                         best_combo_locked = locked
    #                         best_msg = f"FLOW STEER: Exhaustive Cut. Pipe {candidate_idx + 1} (-{drop}). Healed {boosts}x."

    #     if best_combo_sol:
    #         return best_combo_sol, best_combo_locked, best_msg
             
    #     return None, None, "FLOW STEER: Exhaustive search found no valid bypass."

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
        cost, p_min, _, _ = self._get_cached_stats(final_indices)
        p_surplus = p_min - self.simulator.config.h_min
        if p_surplus < 0: return float('inf'), -float('inf'), None
        score = cost - (p_surplus * self.adaptive_bonus)
        return score, cost, squeezed_sol
    
    # def _loop_balancing_kick(self, solution):
    #     """
    #     FLOW STEERING:
    #     Finds all cycles. Tests cutting the largest pipe in each cycle.
    #     Unlike micro-trims, macro-cuts drastically change topology, 
    #     so we MUST run Squeeze on every feasible cut to see the true potential.
    #     """
    #     curr_indices = self._get_current_indices(solution)
    #     _, _, _, crit_node = self._get_cached_stats(curr_indices)
    #     if not crit_node or crit_node == "ERR": return None, None, ""

    #     G_simple = nx.Graph()
    #     edge_to_pipe = {}
    #     for u, v, k in self.simulator.graph.edges(keys=True):
    #         if k in self.simulator.component_names:
    #             idx = self.simulator.component_names.index(k)
    #             G_simple.add_edge(u, v)
    #             edge_to_pipe[(u, v)] = idx
    #             edge_to_pipe[(v, u)] = idx

    #     try: cycles = nx.cycle_basis(G_simple)
    #     except: return None, None, ""
    #     if not cycles: return None, None, "No cycles found"

    #     best_combo_score = float('inf')
    #     best_combo_sol = None
    #     best_combo_locked = set()
    #     best_msg = ""

    #     for cycle_nodes in cycles:
    #         cycle_indices = []
    #         full_cycle = cycle_nodes + [cycle_nodes[0]]
    #         for u, v in zip(full_cycle[:-1], full_cycle[1:]):
    #             if (u, v) in edge_to_pipe:
    #                 cycle_indices.append(edge_to_pipe[(u, v)])

    #         for candidate_idx in cycle_indices:
    #             curr_d_idx = curr_indices[candidate_idx]
    #             if curr_d_idx < 2: continue # Дрібні труби не ріжемо
                
    #             # Залишаємо оптимізацію швидкості: беремо лише найглибший зріз
    #             max_drop = min(3, curr_d_idx) 
    #             target_d_idx = curr_d_idx - max_drop
                
    #             kicked = list(solution)
    #             locked = set()
    #             kicked[candidate_idx] = self.diameters[target_d_idx]
    #             locked.add(candidate_idx)
                
    #             # 1. Швидко лікуємо
    #             healed_sol, is_feasible, boosts = self._heal_network(kicked, locked)
                
    #             if is_feasible:
    #                 # 2. ФІЗИЧНА ВАЛІДАЦІЯ: Робимо Squeeze одразу! Суррогат тут бреше.
    #                 test_squeezed = self._gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=4, quick_mode=True)
    #                 sq_indices = self._get_current_indices(test_squeezed)
    #                 sq_cost, _, _, _ = self._get_cached_stats(sq_indices)
                    
    #                 # Шукаємо абсолютний мінімум РЕАЛЬНОЇ ціни після Сквізу
    #                 if sq_cost < best_combo_score:
    #                     best_combo_score = sq_cost
    #                     # Зберігаємо healed_sol, щоб головний цикл міг зробити глибокий 8-pass Squeeze
    #                     best_combo_sol = healed_sol 
    #                     best_combo_locked = locked
    #                     best_msg = f"FLOW STEER: Cut Pipe {candidate_idx + 1} (-{max_drop}). Healed {boosts}x."

    #     if best_combo_sol:
    #         return best_combo_sol, best_combo_locked, best_msg
             
    #     return None, None, "FLOW STEER: Exhaustive search found no valid bypass."

    def solve_standalone(self):
        self._tabu_fingerprints.clear()
        print("[AnalyticalSolver] 1. Generating Diverse Seeds...")
        seeds = self._make_diverse_seeds()
        active_pool = []
        for s in seeds:
            idx = self._get_current_indices(s)
            c, p, _, _ = self._get_cached_stats(idx)
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
            c, p, _, _ = self._get_cached_stats(idx)
            if c < global_best_cost:
                 diff = global_best_cost - c
                 global_best_cost = c
                 global_best_sol = swapped
                 print(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({global_best_cost/1e6:.4f}M$)")
                 stagnation_counter = 0
                 kick_tabu_set.clear()
                 p_surplus = p - self.simulator.config.h_min
                 score = c - (p_surplus * self.adaptive_bonus)
                 active_pool.insert(0, (score - (global_best_cost * 0.1), c, swapped))

            # FORCE STRATEGY CYCLE
            if stagnation_counter >= 1 or round_idx % 2 == 0:
                # Cycle: TOPO -> BOTTLENECK -> TOPO -> SHOCK
                cycle = round_idx % 6
                if cycle == 0: strategy = "TOPO-DIV"
                elif cycle == 1: strategy = "LOOP_BALANCE"
                elif cycle == 2: strategy = "SYNC_TRIM"     # <--- Наша нова зброя
                elif cycle == 3: strategy = "FINISHER"      
                elif cycle == 4: strategy = "BOTTLENECK"
                else: strategy = "SHOCK"
                
                self._log(f"[FORCE] Applying '{strategy}'...")
                
                pool_sols = [x[2] for x in active_pool[:3]]
                if global_best_sol not in pool_sols:
                    pool_sols.append(global_best_sol)
                import random
                kick_target = random.choice(pool_sols)
                
                forced_sol = None
                locked = None
                path_sig = None
                
                if strategy == "SHOCK":
                    forced_sol, locked, log_msg = self._forcing_hand_kick(kick_target)
                elif strategy == "BOTTLENECK":
                    forced_sol, locked, log_msg = self._upstream_bottleneck_kick(kick_target)
                elif strategy == "LOOP_BALANCE":
                    forced_sol, locked, log_msg = self._loop_balancing_kick(kick_target)
                elif strategy == "FINISHER":
                    forced_sol, locked, log_msg = self._micro_trim_kick(kick_target)
                elif strategy == "SYNC_TRIM":               # <--- Додайте ці два рядки
                    forced_sol, locked, log_msg = self._sync_trim_kick(kick_target)
                else:
                    forced_sol, locked, log_msg, path_sig = self._topological_inversion_kick(kick_target, kick_tabu_set)
                
                if forced_sol and locked:
                    self._log(f"     -> {log_msg}")
                    
                    self._log("     [Phase 1: STARVE - Aggressive Reduction]")
                    starved_sol = self._gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=8, verbose=True)
                    
                    self._log("     [Phase 2: HEAL - Final Polish]")
                    final_sol = self._gradient_squeeze(starved_sol, verbose=True)
                    
                    idx = self._get_current_indices(final_sol)
                    c, p, _, _ = self._get_cached_stats(idx)
                    
                    if p >= self.simulator.config.h_min:
                        p_surplus = p - self.simulator.config.h_min
                        score = c - (p_surplus * self.adaptive_bonus)
                        
                        # --- ФІНАЛЬНЕ ВИПРАВЛЕННЯ: POOL FLUSH ---
                        # Якщо ми застрягли надовго (більше 3 раундів), ми вбиваємо стару "еліту",
                        # щоб дати новому рішенню шанс еволюціонувати, не конкуруючи з 6.11M$
                        if stagnation_counter >= 3:
                            self._log("     [INFO] Flushing active pool to force exploration!")
                            active_pool.clear()
                            
                        # Додаємо нове рішення
                        active_pool.insert(0, (score, c, final_sol))
                        # ----------------------------------------
                        
                        if c < global_best_cost:
                            diff = global_best_cost - c
                            global_best_cost = c
                            global_best_sol = final_sol
                            self._log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({global_best_cost/1e6:.4f}M$)")
                            
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
            for _, _, parent_sol in active_pool:
                curr_indices = self._get_current_indices(parent_sol)
                unit_losses = self._get_cached_heuristics(curr_indices)
                
                # Сортуємо для розширень (де найбільше тертя) та звужень (де найменше)
                high_friction = sorted(range(self.num_pipes), key=lambda i: unit_losses[i], reverse=True)
                low_friction = sorted(range(self.num_pipes), key=lambda i: unit_losses[i]) 
                
                # 1. Поодинокі розширення (рятують тиск)
                for pipe_idx in high_friction[:SINGLE_CANDIDATES]:
                    s, c, sol = self._evaluate_candidate(parent_sol, [pipe_idx], "upgrade")
                    if sol: next_gen.append((s, c, sol))
                
                # 2. Поодинокі звуження (економлять гроші)
                for pipe_idx in low_friction[:4]:
                     s, c, sol = self._evaluate_candidate(parent_sol, [pipe_idx], "downgrade")
                     if sol: next_gen.append((s, c, sol))

                # 3. Парні розширення магістралей (для складних вузлів)
                if len(high_friction) >= 2:
                    for p1, p2 in itertools.combinations(high_friction[:4], 2):
                        s, c, sol = self._evaluate_candidate(parent_sol, [p1, p2], "upgrade")
                        if sol: next_gen.append((s, c, sol))
            
            # Тихий лог завершення пошуку
            self._log(f"   > [R{round_idx+1}] Beam Search completed. Pool expanded.")
            
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
                    real_cost, p_min, _, _ = self._get_cached_stats(idx_sol)
                    
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