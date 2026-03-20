import itertools
import random
import numpy as np
import math

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

    def gradient_squeeze(self, indices, locked_pipes=None, max_passes=None, quick_mode=False, dyn_bonus=None, min_rel_improvement=0.0003):
        if dyn_bonus is None:
            c_start, _, _, _ = self.ctx.get_cached_stats(indices)
            dyn_bonus = c_start * 0.001
            
        locked_pipes = locked_pipes or set()
        current_indices = list(indices)
        improved = True
        passes = 0
        
        cost, p_min, _, _ = self.ctx.get_cached_stats(current_indices)
        best_score = cost - ((p_min - self.ctx.simulator.config.h_min) * dyn_bonus) if p_min >= self.ctx.simulator.config.h_min else float('inf')
        
        milestone_cost = cost
        milestone_pass = 0

        active_indices = [i for i in range(self.ctx.num_pipes) if i not in locked_pipes]

        while improved:
            improved = False
            passes += 1
            
            if getattr(self, 'progress_callback', None):
                self.progress_callback()
                
            if max_passes and passes > max_passes: break
            
            if passes - milestone_pass >= 3:
                rel_improvement = (milestone_cost - cost) / max(milestone_cost, 1.0)
                if rel_improvement < min_rel_improvement:
                    break
                milestone_cost = cost
                milestone_pass = passes

            random.shuffle(active_indices)
          
            active_indices_pass = active_indices
                
            for idx in active_indices_pass:
                curr_d = current_indices[idx]
                best_local_sol = None
                best_local_score = best_score
                best_local_cost = cost
                best_local_p = p_min
                
                if curr_d > 0:
                    test_sol = list(current_indices)
                    test_sol[idx] -= 1
                    c, p_val, feas, _ = self.ctx.get_cached_stats(test_sol)
                    if feas and p_val >= self.ctx.simulator.config.h_min:
                        score = c - ((p_val - self.ctx.simulator.config.h_min) * dyn_bonus)
                        if score < best_local_score or (quick_mode and c < best_local_cost):
                            best_local_score, best_local_sol, best_local_cost, best_local_p = score, test_sol, c, p_val
                            
                if curr_d < self.ctx.max_d_idx and not quick_mode:
                    test_sol = list(current_indices)
                    test_sol[idx] += 1
                    c, p_val, feas, _ = self.ctx.get_cached_stats(test_sol)
                    if feas and p_val >= self.ctx.simulator.config.h_min:
                        score = c - ((p_val - self.ctx.simulator.config.h_min) * dyn_bonus)
                        if score < best_local_score:
                            best_local_score, best_local_sol, best_local_cost, best_local_p = score, test_sol, c, p_val
                            
                if best_local_sol is not None:
                    current_indices = best_local_sol
                    best_score, cost, p_min = best_local_score, best_local_cost, best_local_p
                    improved = True

        return current_indices

    def heal_network(self, kicked, locked_pipes):
        locked_pipes_local = set(locked_pipes) 
        boosts = 0
        
        while True:
            _, min_p, feas, crit_node = self.ctx.get_cached_stats(kicked)
            if feas and min_p >= self.ctx.simulator.config.h_min:
                return kicked, True, boosts
                
            if not crit_node or crit_node == "ERR":
                return kicked, False, boosts
                
            path_pipes, _ = self.ctx.get_dominant_path(kicked, crit_node)
            if not path_pipes:
                return kicked, False, boosts
                
            unit_losses = self.ctx.get_cached_heuristics(kicked)
            candidates = []
            
            for idx in path_pipes:
                if idx in locked_pipes_local: continue 
                curr_d = kicked[idx]
                if curr_d < self.ctx.max_d_idx: 
                    if self.ctx.num_pipes >= 200:
                        c_curr = self.ctx.simulator.costs[curr_d]
                        c_next = self.ctx.simulator.costs[curr_d + 1]
                        delta_c_per_m = c_next - c_curr
                        
                        abs_cost = delta_c_per_m * self.ctx.simulator.lengths[idx]
                        
                        efficiency = unit_losses[idx] / max(1e-6, delta_c_per_m * math.sqrt(max(1.0, abs_cost)))
                        candidates.append((idx, efficiency))
                    else:
                        candidates.append((idx, unit_losses[idx]))
                    
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_pipe = candidates[0][0]
            
            kicked[best_pipe] += 1
            locked_pipes_local.add(best_pipe) 
            boosts += 1
            
        return kicked, False, boosts

    def swap_search(self, indices, dyn_bonus):
        indices_copy = list(indices)
        
        raw_cost, best_p, _, crit_node = self.ctx.get_cached_stats(indices_copy)
        
        if not crit_node or crit_node == "ERR":
            return indices_copy
            
        p_surplus = best_p - self.ctx.simulator.config.h_min
        
        best_cost = raw_cost - (p_surplus * dyn_bonus) if p_surplus >= 0 else float('inf')
            
        path_pipes, _ = self.ctx.get_dominant_path(indices_copy, crit_node)
        unit_losses = self.ctx.get_cached_heuristics(indices_copy)
        lazy_pipes = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i])
        
        if not path_pipes or not lazy_pipes:
            return indices_copy

        up_limit = max(10, self.ctx.num_pipes // 5)
        down_limit = max(10, self.ctx.num_pipes // 3)
        
        if self.ctx.num_pipes > 200:
            down_limit = min(down_limit, 40)
        
        for p in lazy_pipes[:down_limit]:
            if indices_copy[p] > 0:
                test_sol = list(indices_copy)
                test_sol[p] -= 1
                c, p_val, feas, _ = self.ctx.get_cached_stats(test_sol)
                if feas and p_val >= self.ctx.simulator.config.h_min:
                    score = c - ((p_val - self.ctx.simulator.config.h_min) * dyn_bonus)
                    if score < best_cost:
                        best_cost, indices_copy = score, test_sol
                        
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