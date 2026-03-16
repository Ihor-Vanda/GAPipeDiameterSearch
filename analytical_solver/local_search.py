import itertools

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
        if dyn_bonus is None:
            c_start, _, _, _ = self.ctx.get_cached_stats(indices)
            dyn_bonus = c_start * 0.001
        
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
        import random
        kicked = list(indices)
        boosts = 0
        for _ in range(40): 
            _, t_p, t_feas, crit_node = self.ctx.get_cached_stats(kicked)
            if t_feas and t_p >= self.ctx.simulator.config.h_min: return kicked, True, boosts
            
            path_pipes, _ = self.ctx.get_dominant_path(kicked, crit_node)
            if not path_pipes: return kicked, False, boosts
            
            unit_losses = self.ctx.get_cached_heuristics(kicked)
            
            candidates = []
            for idx in path_pipes:
                if idx in locked_pipes: continue 
                curr_d = kicked[idx]
                if curr_d < self.ctx.max_d_idx: 
                    # 🔴 АДАПТАЦІЯ: Економічне зцілення тільки для великих мереж
                    if self.ctx.num_pipes >= 200:
                        penalty = (curr_d + 1) * self.ctx.simulator.lengths[idx]
                        efficiency = (unit_losses[idx] * 1000.0) / max(1.0, penalty)
                        candidates.append((idx, efficiency))
                    else:
                        # Для малих мереж (Ханой) - чиста жадібна гідравліка (б'ємо по найгіршому)
                        candidates.append((idx, unit_losses[idx]))
                    
            if not candidates:
                break
                
            # Сортуємо за ЕФЕКТИВНІСТЮ (Користь / Вартість), а не просто за втратою тиску
            candidates.sort(key=lambda x: x[1], reverse=True)
            pool_size = min(4, len(candidates))
            
            # Рандомізація вибору з Топ-4 найефективніших труб
            weights = [0.5, 0.25, 0.15, 0.1][:pool_size]
            worst_pipe = random.choices(candidates[:pool_size], weights=weights)[0][0]
            
            kicked[worst_pipe] += 1
            boosts += 1
            
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