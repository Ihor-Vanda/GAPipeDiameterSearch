import math
import random
import time
import bisect
import itertools
import multiprocessing
import networkx as nx

from .context import SolverContext
from .pool import SolutionPool
from .local_search import LocalSearch
from .kicks import KickStrategies

class AnalyticalSolver:
    worker_task = None 
    
    def __init__(self, simulator_instance, available_diameters, v_opt=1.0, max_sims=None, 
                 time_limit_sec=None, pool=None, log_dir=None, n_workers=1):
        
        self.ctx = SolverContext(simulator_instance, available_diameters, v_opt)
        self.ls = LocalSearch(self.ctx)
        self.kicker = KickStrategies(self.ctx, self.ls)
        
        self.mp_pool = pool
        self.pool = SolutionPool(self.ctx)
        
        self.log_dir = log_dir
        self.n_workers = n_workers
        
        n = self.ctx.num_pipes
        if n < 50: self.network_class = "SMALL"
        elif n < 200: self.network_class = "MEDIUM"
        elif n < 1000: self.network_class = "LARGE"
        else: self.network_class = "XLARGE"
            
        self._configure_for_scale()
        
        if max_sims is not None:
            self.max_sims = max_sims
        else:
            self.max_sims = self.BASE_SIM_BUDGET
            
        if time_limit_sec is not None:
            self.time_limit_sec = time_limit_sec
        else:
            secs_needed = self.max_sims / max(self.ctx.sim_speed, 1.0)
            self.time_limit_sec = max(3600, secs_needed * 2.0)
            
        print(f"[Config] Network: {self.network_class} ({self.ctx.num_pipes} pipes)")
        print(f"  Target Budget: {self.max_sims:,} sims (≈{self.time_limit_sec/60:.1f} min)")
        print(f"  BeamWidth={self.BEAM_WIDTH}")
        
        self.strategies = ["TOPO-DIV", "LOOP_BALANCE", "SYNC_TRIM", "FINISHER", "BOTTLENECK", "SHOCK"]
        if nx.is_tree(self.ctx.base_G_flow):
            self.strategies.remove("LOOP_BALANCE")
        self.strategies.append("DIAM_DIVERSITY")
        self.strategies.append("RUIN_RECREATE")
        
        self.strat_wins = {s: 1.0 for s in self.strategies}
        self.strat_tries = {s: 1 for s in self.strategies}

    def _configure_for_scale(self):
        n = self.ctx.num_pipes
        cls = self.network_class
        
        self.BEAM_WIDTH = {"SMALL": max(5, n//10), "MEDIUM": max(5, n//20), "LARGE": 8, "XLARGE": 6}[cls]
        self.SINGLE_CANDIDATES = {"SMALL": max(6, n//5), "MEDIUM": max(6, n//10), "LARGE": max(6, n//20), "XLARGE": 10}[cls]
        self.BASE_SIM_BUDGET = {"SMALL": 30000, "MEDIUM": 100000, "LARGE": 1000000, "XLARGE": 3000000}[cls]

    def _calculate_ideal_d(self, flow, target_v):
        if abs(flow) < 1e-6: return 0
        d_ideal = math.sqrt((4.0 * abs(flow)) / (math.pi * target_v))
        pos = bisect.bisect_left(self.ctx.diameters, d_ideal)
        if pos < len(self.ctx.diameters): return pos
        return self.ctx.max_d_idx

    def _make_reserve_pool(self, size=4):
        reserve = []
        for _ in range(size):
            seed = [random.randint(0, self.ctx.max_d_idx) for _ in range(self.ctx.num_pipes)]
            healed, ok, _ = self.ls.heal_network(seed, set())
            if ok:
                n_freeze = max(3, self.ctx.num_pipes // 4)
                frozen = set(random.sample(range(self.ctx.num_pipes), n_freeze))
                squeezed = self.ls.gradient_squeeze(healed, locked_pipes=frozen, quick_mode=True)
                reserve.append(squeezed)
        return reserve

    def _make_diverse_seeds(self):
        seeds = []
        for v in [1.0, 1.2, 0.8]:
            idx_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            for _ in range(10):
                real_diams = [self.ctx.diameters[i] for i in idx_sol]
                flows, _ = self.ctx.simulator.get_hydraulic_state(real_diams)
                new_idx = [self._calculate_ideal_d(q, v) for q in flows] 
                if new_idx == idx_sol: break
                idx_sol = new_idx
                
            _, p, feas, _ = self.ctx.get_cached_stats(idx_sol)
            if not feas or p < self.ctx.simulator.config.h_min:
                idx_sol, is_healed, _ = self.ls.heal_network(idx_sol, set())
                if not is_healed: continue 
            squeezed_sol = self.ls.gradient_squeeze(idx_sol, quick_mode=True)
            seeds.append(squeezed_sol)
        return seeds
    
    def _make_relinked_seeds(self, archive):
        seeds = []
        if len(archive) < 2: return []

        top_sols = [x[1] for x in archive[:4]]
        pairs = list(itertools.combinations(top_sols, 2))
        random.shuffle(pairs)

        max_steps = {"SMALL": 999, "MEDIUM": 30, "LARGE": 15, "XLARGE": 10}[self.network_class]

        for sol_A, sol_B in pairs:
            diff_indices = [i for i in range(self.ctx.num_pipes) if sol_A[i] != sol_B[i]]
            current_path_sol = list(sol_A)
            
            steps_taken = 0
            while diff_indices and len(seeds) < self.BEAM_WIDTH and steps_taken < max_steps:
                steps_taken += 1
                candidates = diff_indices if len(diff_indices) <= 10 else random.sample(diff_indices, 10)
                
                best_idx, best_score = -1, float('inf')
                best_feas, best_p = False, 0
                
                for idx in candidates:
                    test_sol = list(current_path_sol)
                    test_sol[idx] = sol_B[idx]
                    c, p, feas, _ = self.ctx.get_cached_stats(test_sol)
                    score = c if (feas and p >= self.ctx.simulator.config.h_min) else c * 1.5
                    
                    if score < best_score:
                        best_score, best_idx = score, idx
                        best_feas, best_p = feas, p
                        
                current_path_sol[best_idx] = sol_B[best_idx]
                diff_indices.remove(best_idx)
                
                if best_feas and best_p >= self.ctx.simulator.config.h_min:
                    squeezed = self.ls.gradient_squeeze(current_path_sol, quick_mode=True)
                    seeds.append(squeezed)
                    
            if len(seeds) >= self.BEAM_WIDTH: break
                
        while len(seeds) < self.BEAM_WIDTH:
            seeds.append(list(top_sols[0]))
        return seeds

    def _make_warm_seeds(self, archive, worker_id=0, failed_basins=None):
        failed_basins = failed_basins or set()
        
        if not archive:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🏴‍☠️ ADVENTURER (Empty Archive Fallback)")
            return self._make_diverse_seeds()
        
        seeds = []
        if worker_id in [3, 7, 11, 15, 19]: role = "ADVENTURER"
        elif worker_id == 0: role = "EXPLOITER"
        elif worker_id == 1: role = "RELINKER"
        elif worker_id == 2: role = "EXPLORER"
        else:
            caste_roll = random.random()
            if caste_roll < 0.30: role = "EXPLOITER"
            elif caste_roll < 0.60: role = "RELINKER"
            elif caste_roll < 0.85: role = "EXPLORER"
            else: role = "ADVENTURER"
            
        if role == "RELINKER" and len(archive) >= 2:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🧬 RELINKER (Path-Relinking between Top-4)")
            return self._make_relinked_seeds(archive)
        elif role == "EXPLOITER" or (role == "RELINKER" and len(archive) < 2):
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🎯 EXPLOITER (Micro-mutations on Top-2)")
            top_sols = [x[1] for x in archive[:2]]
            n_perturb = max(1, self.ctx.num_pipes // 15) 
            deltas = [-1, 1]                             
        elif role == "EXPLORER":
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🔭 EXPLORER (Macro-mutations on Top-5)")
            top_sols = [x[1] for x in archive]
            n_perturb = max(3, self.ctx.num_pipes // 8)  
            deltas = [-2, -1, 1, 2]                      
        else:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🏴‍☠️ ADVENTURER (Cold Start - Ignored Archive)")
            return self._make_diverse_seeds()
            
        for _ in range(self.BEAM_WIDTH):
            # 🔴 ОНОВЛЕНО: Basin Avoidance (До 10 спроб знайти унікальний старт)
            for _ in range(10):
                base_sol = random.choice(top_sols)
                perturbed = list(base_sol)
                pipes_to_shake = random.sample(range(self.ctx.num_pipes), n_perturb)
                
                for p in pipes_to_shake:
                    perturbed[p] = max(0, min(self.ctx.max_d_idx, perturbed[p] + random.choice(deltas)))
                    
                healed_sol, is_feas, _ = self.ls.heal_network(perturbed, set())
                if is_feas:
                    squeezed = self.ls.gradient_squeeze(healed_sol, quick_mode=True)
                    if self.pool.get_basin_signature(squeezed) not in failed_basins:
                        seeds.append(squeezed)
                        break # Успіх! Знайшли нову долину
            else:
                # Якщо за 10 спроб не вийшли з ями, беремо як є
                seeds.append(base_sol) 
            
        return seeds

    def _run_single_search(self, seeds, time_budget, global_best_cost, worker_id=0, shared_progress=None):
        if not seeds:
            seeds = [[self.ctx.max_d_idx] * self.ctx.num_pipes]
            
        run_best_cost = float('inf')
        run_best_sol = None
        
        start_time = time.time()
        self.pool.clear_all()
        
        reserve_pool = self._make_reserve_pool(size=4)
        
        best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds])
        base_dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001 
        
        valid_sols = []
        for s in seeds:
            c, p, feas, _ = self.ctx.get_cached_stats(s)
            p_surplus = p - self.ctx.simulator.config.h_min
            score = c - (p_surplus * base_dyn_bonus)
            self.pool.active_pool.append((score, c, s))
            self.pool.add_to_tabu(s, c)
            if feas and p >= self.ctx.simulator.config.h_min:
                valid_sols.append((c, s))
                
        if valid_sols:
            run_best_cost = min(x[0] for x in valid_sols)
            run_best_sol = next(x[1] for x in valid_sols if x[0] == run_best_cost)
        else:
            run_best_cost = global_best_cost
            run_best_sol = min(self.pool.active_pool, key=lambda x: x[0])[2] 

        stagnation_counter = 0
        self.ctx.log(f"   > Seeds initialized. Baseline Target: {run_best_cost/1e6:.4f}M$")

        round_idx = 0
        last_published_cost = float('inf')
        epoch_start_sims = self.ctx.sim_count 
        
        while True:
            self.pool.current_round = round_idx 
            
            jitter_factor = random.uniform(0.70, 1.30)
            base_dyn_bonus = min(run_best_cost, global_best_cost) * 0.001 * jitter_factor

            if shared_progress is not None:
                gb = shared_progress.get('global_best')
                if gb and gb[0] < global_best_cost:
                    global_best_cost = gb[0]
                    self.ctx.log(f"   > [IPC] 📡 Received new Global Bound from peer: {global_best_cost/1e6:.4f}M$")

            if shared_progress is not None and worker_id is not None:
                shared_progress[worker_id] = {"round": round_idx + 1, "sims": self.ctx.sim_count, "best_cost": run_best_cost}
            
            elapsed = time.time() - start_time
            epoch_sims = self.ctx.sim_count - epoch_start_sims
            
            if elapsed > time_budget or epoch_sims >= self.max_sims: break
                
            just_flushed = False
            progress_ratio = min(1.0, epoch_sims / max(1, self.max_sims)) 
            is_late_game = progress_ratio > 0.5 
            stag_limit = 4 + int(4 * progress_ratio) 
            
            if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()
            next_gen = []
            
            if round_idx % 3 == 0 or stagnation_counter == 0:
                swapped = self.ls.swap_search(run_best_sol, base_dyn_bonus)
                c, p, feas, _ = self.ctx.get_cached_stats(swapped)
                if feas and p >= self.ctx.simulator.config.h_min and c < run_best_cost:
                    is_ghost = False
                    if (run_best_cost - c) > (run_best_cost * 0.02):
                        is_ghost = self.ctx.is_ghost_solution(swapped, c)
                        
                    if is_ghost:
                        self.ctx.log(f"   > [SHIELD] Swap illusion blocked ({c/1e6:.4f}M$)!")
                    else:
                         diff = run_best_cost - c
                         run_best_cost, run_best_sol = c, swapped
                         self.ctx.log(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$)")
                         
                         if run_best_cost < last_published_cost:
                             if shared_progress is not None:
                                 shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                             last_published_cost = run_best_cost

                         if run_best_cost < global_best_cost:
                             global_best_cost = run_best_cost
                             if shared_progress is not None:
                                 current_gb = shared_progress.get('global_best', (float('inf'), []))
                                 if run_best_cost < current_gb[0]:
                                     shared_progress['global_best'] = (run_best_cost, list(run_best_sol))

                         stagnation_counter = 0
                         self.pool.kick_tabu_set.clear()
                         
                         p_surplus = p - self.ctx.simulator.config.h_min
                         eff_bonus = base_dyn_bonus * 0.3 if p_surplus < 1.0 else (base_dyn_bonus * 2.0 if p_surplus > 15.0 else base_dyn_bonus)
                         self.pool.active_pool.insert(0, (c - (p_surplus * eff_bonus) - (run_best_cost * 0.1), c, swapped))

            if stagnation_counter >= 1 or round_idx % 2 == 0:
                n_total = sum(self.strat_tries.values())
                
                peer_archive = []
                if shared_progress is not None:
                    if run_best_cost < last_published_cost:
                        shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                        last_published_cost = run_best_cost
                        
                    for i in range(self.n_workers):
                        if i != worker_id:
                            peer_data = shared_progress.get(f'best_sol_{i}')
                            if peer_data: peer_archive.append(peer_data)

                peer_to_cross = None
                if stagnation_counter >= stag_limit * 3:
                    strategy = "SEGMENT_RESTART"
                    self.pool.add_basin_to_tabu(run_best_sol) 
                elif stagnation_counter >= stag_limit * 2 and peer_archive:
                    corridor_peers = [p for p in peer_archive if 30 <= self.pool.hamming_distance(run_best_sol, p[1]) <= 70]
                    diverse_peers = [p for p in peer_archive if self.pool.hamming_distance(run_best_sol, p[1]) > 70]
                    
                    if corridor_peers:
                        strategy = "CORRIDOR_SEARCH"
                        peer_to_cross = random.choice(corridor_peers)
                    elif diverse_peers:
                        strategy = "IPC_CROSSOVER"
                        peer_to_cross = min(diverse_peers, key=lambda x: x[0])
                    else:
                        strategy = "ILS_PERTURBATION"
                elif stagnation_counter >= stag_limit * 1.5:
                    strategy = "ILS_PERTURBATION"
                elif stagnation_counter > 0 and stagnation_counter % 12 == 0:
                    strategy = "RUIN_RECREATE"
                else:
                    valid_strats = [s for s in self.strategies if s not in ["SEGMENT_RESTART", "ILS_PERTURBATION", "IPC_CROSSOVER", "CORRIDOR_SEARCH"]]
                    strategy = max(valid_strats, key=lambda s: (self.strat_wins[s] / self.strat_tries.get(s, 1)) + math.sqrt(2 * math.log(max(1, n_total)) / self.strat_tries.get(s, 1)))
                
                self.strat_tries[strategy] = self.strat_tries.get(strategy, 0) + 1
                self.ctx.log(f"[FORCE] Escalation Level: {stagnation_counter}/{stag_limit} -> Applying '{strategy}'...")
                
                if stagnation_counter >= stag_limit and self.pool.active_pool:
                    self.ctx.log("     [DIVERSITY] Stagnation high. Kicking the most diverse solution in pool.")
                    kick_target = max(
                        [x[2] for x in self.pool.active_pool],
                        key=lambda s: self.pool.hamming_distance(s, run_best_sol)
                    )
                else:
                    kick_mode = round_idx % 3
                    if kick_mode == 0: kick_target = run_best_sol
                    elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
                    else:
                        pool_sols = [x[2] for x in self.pool.active_pool[:3]]
                        kick_target = random.choice(pool_sols) if pool_sols else run_best_sol
                
                forced_sol, locked, path_sig = None, None, None
                
                if strategy == "SEGMENT_RESTART": forced_sol, locked, log_msg = self.kicker.segment_restart_kick(kick_target, base_dyn_bonus)
                elif strategy == "IPC_CROSSOVER": forced_sol, locked, log_msg = self.kicker.crossover_with_peer_kick(kick_target, peer_to_cross[1], run_best_cost, peer_to_cross[0])
                elif strategy == "CORRIDOR_SEARCH": forced_sol, locked, log_msg = self.kicker.corridor_search_kick(kick_target, peer_to_cross[1])
                elif strategy == "ILS_PERTURBATION": forced_sol, locked, log_msg = self.kicker.ils_perturbation_kick(kick_target, stagnation_counter)
                elif strategy == "SHOCK": forced_sol, locked, log_msg = self.kicker.forcing_hand_kick(kick_target)
                elif strategy == "BOTTLENECK": forced_sol, locked, log_msg = self.kicker.upstream_bottleneck_kick(kick_target)
                elif strategy == "LOOP_BALANCE": forced_sol, locked, log_msg = self.kicker.loop_balancing_kick(kick_target, base_dyn_bonus)
                elif strategy == "DIAM_DIVERSITY": forced_sol, locked, log_msg = self.kicker.diameter_diversity_kick(kick_target, stagnation_counter) 
                elif strategy == "RUIN_RECREATE": forced_sol, locked, log_msg = self.kicker.ruin_and_recreate_kick(kick_target, stagnation_counter) 
                elif strategy == "FINISHER": forced_sol, locked, log_msg = self.kicker.micro_trim_kick(kick_target)
                elif strategy == "SYNC_TRIM": forced_sol, locked, log_msg = self.kicker.sync_trim_kick(kick_target, base_dyn_bonus)
                else: forced_sol, locked, log_msg, path_sig = self.kicker.topological_inversion_kick(kick_target, self.pool.kick_tabu_set)
                
                if forced_sol and locked:
                    self.ctx.log(f"     -> {log_msg}")
                    
                    if strategy == "SEGMENT_RESTART":
                        final_sol = forced_sol 
                    elif strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "CORRIDOR_SEARCH"]:
                        # 🔴 ПАТЧ 2: М'яка посадка. max_passes=2 замість None. 
                        # Даємо пулу шанс дослідити "схил" нової долини, перш ніж жадібний пошук скотить його на дно.
                        final_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=set(), max_passes=2, quick_mode=True, dyn_bonus=base_dyn_bonus)
                    elif strategy == "RUIN_RECREATE":
                        final_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=4, quick_mode=True, dyn_bonus=base_dyn_bonus)
                    else:
                        starved_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked, max_passes=8, dyn_bonus=base_dyn_bonus)
                        final_sol = self.ls.gradient_squeeze(starved_sol, dyn_bonus=base_dyn_bonus)
                    
                    c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                    
                    if feas and p >= self.ctx.simulator.config.h_min:
                        p_surplus = p - self.ctx.simulator.config.h_min
                        eff_bonus = base_dyn_bonus * 0.3 if p_surplus < 1.0 else (base_dyn_bonus * 2.0 if p_surplus > 15.0 else base_dyn_bonus)
                        
                        if p_surplus < 2.0:
                            final_sol = self.ls.gradient_squeeze(final_sol, locked_pipes=set(), max_passes=3, quick_mode=True, dyn_bonus=eff_bonus)
                            c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                            p_surplus = p - self.ctx.simulator.config.h_min if feas else p_surplus
                            eff_bonus = base_dyn_bonus * 0.3 if p_surplus < 1.0 else base_dyn_bonus
                        
                        score = c - (p_surplus * eff_bonus)
                        # 🔴 ПАТЧ 1: Tidal Water Level. Чим довше стагнація, тим вище піднімається вода (до +5%), 
                        # що дозволяє алгоритму "перелитись" через край локальної ями.
                        stagnation_relax = min(0.05, (stagnation_counter / max(1, stag_limit)) * 0.015)
                        water_level = global_best_cost * (1.05 - 0.05 * progress_ratio + stagnation_relax)
                        
                        if c < run_best_cost:
                            is_ghost = False
                            if (run_best_cost - c) > (run_best_cost * 0.02):
                                is_ghost = self.ctx.is_ghost_solution(final_sol, c)
                                
                            if is_ghost:
                                self.ctx.log(f"   > [SHIELD] Force illusion blocked ({c/1e6:.4f}M$)!")
                            else:
                                diff = run_best_cost - c
                                run_best_cost, run_best_sol = c, final_sol
                                self.pool.active_pool.insert(0, (score, c, final_sol))
                                stagnation_counter = 0 
                                
                                if strategy in self.strategies: self.strat_wins[strategy] += 3.0 
                                self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$)")
                                
                                if run_best_cost < last_published_cost:
                                    if shared_progress is not None:
                                        shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                                    last_published_cost = run_best_cost

                                if run_best_cost < global_best_cost:
                                    global_best_cost = run_best_cost
                                    if shared_progress is not None:
                                        current_gb = shared_progress.get('global_best', (float('inf'), []))
                                        if run_best_cost < current_gb[0]:
                                            shared_progress['global_best'] = (run_best_cost, list(run_best_sol))

                        elif c < water_level and not self.pool.is_basin_tabu(final_sol):
                            self.pool.active_pool.append((score * 1.05, c, final_sol))
                            if strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "SEGMENT_RESTART", "CORRIDOR_SEARCH", "RUIN_RECREATE"]:
                                stagnation_counter = 0 
                            if strategy in self.strategies: self.strat_wins[strategy] += 0.5
                            self.ctx.log(f"     -> Added to Pool (Water Level Accept): {c/1e6:.4f}M$")
                        else:
                            self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {c/1e6:.4f}M$")
                            if strategy == "SEGMENT_RESTART":
                                stagnation_counter = int(stag_limit * 1.5) # Знижуємо ескалацію
                            
                        if path_sig: self.pool.kick_tabu_set.add(path_sig)
                    else:
                         self.ctx.log(f"     -> Injection Failed: Infeasible (P={p:.2f}m)")

            for _, _, parent_sol in self.pool.active_pool:
                unit_losses = self.ctx.get_cached_heuristics(parent_sol)
                high_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i], reverse=True)
                low_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i]) 
                
                _, p, feas, _ = self.ctx.get_cached_stats(parent_sol)
                parent_p_surplus = (p - self.ctx.simulator.config.h_min) if feas else 0.0
                
                if parent_p_surplus > 10.0:
                    downgrade_limit = 15
                    upgrade_limit = self.SINGLE_CANDIDATES
                elif parent_p_surplus < 2.0:
                    downgrade_limit = 3 
                    upgrade_limit = self.SINGLE_CANDIDATES // 2
                else:
                    downgrade_limit = 8
                    upgrade_limit = self.SINGLE_CANDIDATES
                
                if self.network_class in ("LARGE", "XLARGE"):
                    top_k = max(20, self.ctx.num_pipes // 5)
                    focus_pipes = set(self.ls.get_high_impact_pipes(parent_sol, top_k))
                    high_friction = [p for p in high_friction if p in focus_pipes]
                    low_friction = [p for p in low_friction if p in focus_pipes]
                
                for pipe_idx in high_friction[:upgrade_limit]:
                    s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "upgrade", base_dyn_bonus)
                    if sol: next_gen.append((s, c, sol))
                
                for pipe_idx in low_friction[:downgrade_limit]:
                     s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "downgrade", base_dyn_bonus)
                     if sol: next_gen.append((s, c, sol))
                     
                if len(high_friction) >= 2:
                    combo_limit = 5 if parent_p_surplus >= 2.0 else 3
                    for p1, p2 in itertools.combinations(high_friction[:combo_limit], 2): 
                        s, c, sol = self.ls.evaluate_candidate(parent_sol, [p1, p2], "upgrade", base_dyn_bonus)
                        if sol: next_gen.append((s, c, sol))
            
            if next_gen:
                next_gen.sort(key=lambda x: x[0]) 
                
            unique_next_pool = []
            found_new_record = False
            
            max_d = max(2, self.ctx.num_pipes // 7) 
            min_dist = max(1, int(max_d * (1.0 - (progress_ratio ** 2))))
            
            for rank, (score, cost, sol) in enumerate(next_gen):
                if self.pool.is_tabu(sol, cost): continue
                if len(unique_next_pool) < self.BEAM_WIDTH:
                    if rank == 0: refined_sol = self.ls.gradient_squeeze(sol, dyn_bonus=base_dyn_bonus)
                    elif rank < 3: refined_sol = self.ls.gradient_squeeze(sol, max_passes=4, quick_mode=not is_late_game, dyn_bonus=base_dyn_bonus)
                    else: refined_sol = sol 

                    is_diverse = True
                    for _, _, peer_sol in unique_next_pool:
                        if self.pool.hamming_distance(refined_sol, peer_sol) < min_dist:
                            is_diverse = False; break
                            
                    if not is_diverse and not just_flushed: continue

                    real_cost, p_min, feas_min, _ = self.ctx.get_cached_stats(refined_sol)
                    if feas_min and p_min >= self.ctx.simulator.config.h_min:
                        real_p_surplus = p_min - self.ctx.simulator.config.h_min
                        eff_bonus = base_dyn_bonus * 0.3 if real_p_surplus < 1.0 else (base_dyn_bonus * 2.0 if real_p_surplus > 15.0 else base_dyn_bonus)
                        
                        unique_next_pool.append((real_cost - (real_p_surplus * eff_bonus), real_cost, refined_sol))
                        self.pool.add_to_tabu(refined_sol, real_cost)
                        
                        if real_cost < run_best_cost:
                            is_ghost = False
                            if (run_best_cost - real_cost) > (run_best_cost * 0.02):
                                is_ghost = self.ctx.is_ghost_solution(refined_sol, real_cost)
                                
                            if is_ghost:
                                self.ctx.log(f"   > [SHIELD] Beam illusion blocked ({real_cost/1e6:.4f}M$).")
                                continue 

                            diff = run_best_cost - real_cost
                            run_best_cost, run_best_sol = real_cost, refined_sol
                            found_new_record = True
                            self.ctx.log(f"   > [R{round_idx+1}] 💎 New Record: -${diff:,.0f} ({run_best_cost/1e6:.4f}M$)")
                            
                            if run_best_cost < last_published_cost:
                                if shared_progress is not None:
                                    shared_progress[f'best_sol_{worker_id}'] = (run_best_cost, list(run_best_sol))
                                last_published_cost = run_best_cost

                            if run_best_cost < global_best_cost:
                                global_best_cost = run_best_cost
                                if shared_progress is not None:
                                    current_gb = shared_progress.get('global_best', (float('inf'), []))
                                    if run_best_cost < current_gb[0]:
                                        shared_progress['global_best'] = (run_best_cost, list(run_best_sol))

            if unique_next_pool:
                unique_next_pool.sort(key=lambda x: x[0])
                
                dynamic_beam = max(3, int(self.BEAM_WIDTH * (1.0 + 0.5 * (1.0 - progress_ratio))))
                self.pool.active_pool = unique_next_pool[:dynamic_beam]
                
                if found_new_record or just_flushed:
                    stagnation_counter = 0
                    self.pool.kick_tabu_set.clear()
                else:
                    stagnation_counter += 1
                    if stagnation_counter % 8 == 0: self.pool.kick_tabu_set.clear()
                    
                if len(self.pool.active_pool) >= 3:
                    sols = [x[2] for x in self.pool.active_pool]
                    pairs = list(itertools.combinations(range(len(sols)), 2))
                    avg_dist = sum(self.pool.hamming_distance(sols[a], sols[b]) for a, b in pairs) / len(pairs)
                    
                    diversity_threshold = self.ctx.num_pipes // 8
                    if avg_dist < diversity_threshold:
                        if not reserve_pool:
                            self.ctx.log(f"   [DIVERSITY] Regenerating reserve pool...")
                            reserve_pool = self._make_reserve_pool(size=2)
                        
                        if reserve_pool:
                            self.ctx.log(f"   [DIVERSITY] Pool homogenizing (dist={avg_dist:.1f}). Injecting reserve seed!")
                            fresh = reserve_pool.pop(0) 
                            c, p, feas, _ = self.ctx.get_cached_stats(fresh)
                            if feas:
                                p_surplus = p - self.ctx.simulator.config.h_min
                                score = c - (p_surplus * base_dyn_bonus)
                                self.pool.active_pool[-1] = (score, c, fresh) 
            else:
                self.ctx.log("     [EMERGENCY] Beam search deadlocked (no valid/diverse children). Flushing pool!")
                self.pool.active_pool.clear()
                stagnation_counter += 1

            # 🔴 ПАТЧ 3: Жорсткий Аварійний Респаун (Безумовне прийняття в пул)
            if not self.pool.active_pool:
                self.ctx.log("     [EMERGENCY] Pool empty! Forcing ILS respawn...")
                forced_rescue, locked_rescue, _ = self.kicker.ils_perturbation_kick(run_best_sol, 15)
                if forced_rescue is not None:
                    c, p, feas, _ = self.ctx.get_cached_stats(forced_rescue)
                    if feas and p >= self.ctx.simulator.config.h_min:
                        # Штучно робимо score ідеальним, щоб воно гарантовано прижилося в пулі
                        self.pool.active_pool.append((c - 1e6, c, forced_rescue))
                        stagnation_counter = 0 
                else:
                    safe_seed = [random.randint(0, self.ctx.max_d_idx) for _ in range(self.ctx.num_pipes)]
                    healed_seed, ok, _ = self.ls.heal_network(safe_seed, set())
                    if ok:
                        c, p, feas, _ = self.ctx.get_cached_stats(healed_seed)
                        if feas and p >= self.ctx.simulator.config.h_min:
                            self.pool.active_pool.append((c - 1e6, c, healed_seed))
                            stagnation_counter = 0

            round_idx += 1
                
        if shared_progress is not None and worker_id is not None:
            shared_progress[worker_id] = {"round": "DONE", "sims": self.ctx.sim_count, "best_cost": run_best_cost}

        return run_best_cost, run_best_sol

    def solve_standalone(self):
        print("\n[AnalyticalSolver] ⚡ Initiating Island Model Search...\n")
        start_time = time.time()
        global_best_cost = float('inf')
        global_best_sol = None
        global_archive = []

        epochs = {"SMALL": 2, "MEDIUM": 2, "LARGE": 4, "XLARGE": 4}[self.network_class]
        time_per_epoch = self.time_limit_sec / epochs

        if self.mp_pool:
            manager = multiprocessing.Manager()
            shared_progress = manager.dict()
            for i in range(self.n_workers):
                shared_progress[i] = 0
        else:
            shared_progress = None

        # 🔴 ОНОВЛЕНО: Глобальний трекер мертвих долин
        global_failed_basins = set()

        for epoch in range(epochs):
            mode_str = "PARALLEL" if self.mp_pool else "SEQUENTIAL"
            print("="*46)
            print(f" [EPOCH {epoch+1}/{epochs}] {mode_str} Workers: {self.n_workers} | Time Limit: {time_per_epoch/60:.1f} min")
            print("="*46)
            
            seed_modifier = random.randint(1, 10000)
            tasks = []
            for i in range(self.n_workers):
                # 🔴 ОНОВЛЕНО: Передаємо global_failed_basins у таску
                tasks.append((
                    self.ctx.diameters, self.ctx.v_opt, time_per_epoch, 
                    global_best_cost, global_archive, 
                    seed_modifier + i, i, shared_progress, self.log_dir, epoch,
                    global_failed_basins
                ))

            epoch_results = []
            
            if self.mp_pool:
                async_results = []
                for t in tasks:
                    res = self.mp_pool.apply_async(self.worker_task, (t,))
                    async_results.append((t[6], res))

                epoch_start_time = time.time()
                last_print_time = 0
                print_interval = {"SMALL": 30.0, "MEDIUM": 60.0, "LARGE": 180.0, "XLARGE": 300.0}[self.network_class]
                
                while True:
                    all_done = all(res.ready() for _, res in async_results)
                    if all_done: break
                        
                    curr_time = time.time()
                    if curr_time - last_print_time >= print_interval:
                        last_print_time = curr_time
                        elapsed_total = curr_time - start_time 
                        m, s = divmod(int(elapsed_total), 60)
                        
                        status_parts = []
                        total_sims = 0
                        live_best = global_best_cost
                        
                        for wid in range(self.n_workers):
                            prog = shared_progress.get(wid, 0)
                            
                            if isinstance(prog, dict):
                                sims = prog.get('sims', 0)
                                w_best = prog.get('best_cost', float('inf'))
                                if w_best < live_best: live_best = w_best
                            else:
                                sims = prog
                                
                            total_sims += sims
                            
                            if sims > 0: status_parts.append(f"W{wid+1}:{sims//1000}k")
                            else: status_parts.append(f"W{wid+1}:--")
                                
                        status_str = " ".join(status_parts)
                        best_str = f"{live_best/1e6:.4f}M$" if live_best != float('inf') else "---"
                        print(f"   > [Live {m:02d}:{s:02d}] Best: {best_str} | Sims: {total_sims/1000:.1f}k | {status_str}")
                    
                    time.sleep(1.0)

                for wid, res in async_results:
                    try:
                        # 🔴 ОНОВЛЕНО: Розпаковуємо worker_basins та оновлюємо глобальну пам'ять
                        c, sol, _, sims_done, worker_basins = res.get()
                        if sol is not None: epoch_results.append((c, sol))
                        global_failed_basins.update(worker_basins)
                    except Exception as e:
                        print(f"     [Error] Worker {wid+1} crashed: {e}")

            else:
                for t in tasks:
                    try:
                        c, sol, _, sims_done, worker_basins = self.worker_task(t)
                        if sol is not None: epoch_results.append((c, sol))
                        global_failed_basins.update(worker_basins)
                        print(f"   > Worker {t[6]+1} Finished. Best: {c/1e6:.4f}M$")
                    except Exception as e:
                        print(f"     [Error] Sequential Worker {t[6]+1} crashed: {e}")

            if epoch_results:
                epoch_results_sorted = sorted(epoch_results, key=lambda x: x[0])
                best_epoch_c, best_epoch_sol = epoch_results_sorted[0]
                
                if best_epoch_c < global_best_cost:
                    global_best_cost = best_epoch_c
                    global_best_sol = best_epoch_sol
                    print(f"\n 🏆 [EPOCH {epoch+1}] NEW GLOBAL BEST: {global_best_cost/1e6:.4f}M$ 🏆\n")

                elite_archive = [epoch_results_sorted[0]]
                diverse_archive = []
                req_dist = self.ctx.num_pipes // 10

                for cost, sol in epoch_results_sorted[1:]:
                    min_dist = min(
                        (sum(a != b for a, b in zip(sol, s)) for _, s in elite_archive + diverse_archive),
                        default=float('inf')
                    )
                    if min_dist > req_dist and len(diverse_archive) < 4:
                        diverse_archive.append((cost, sol))
                    elif len(elite_archive) < 2 and min_dist <= req_dist:
                        elite_archive.append((cost, sol))

                global_archive = elite_archive + diverse_archive

        print("\n[FINAL POLISH] Polishing global best solution...")
        if global_best_sol:
            polished = self.ls.gradient_squeeze(global_best_sol, max_passes=None, quick_mode=False, dyn_bonus=global_best_cost * 0.001)
            p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
            
            if p_p >= self.ctx.simulator.config.h_min and p_cost < global_best_cost:
                global_best_cost = p_cost
                global_best_sol = polished
                print(f"   > [POLISH] Improved! Final: {global_best_cost/1e6:.4f}M$")
        else:
            print("\n[WARNING] No valid solution found. Returning safe default.")
            global_best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            try: global_best_cost, _, _, _ = self.ctx.get_cached_stats(global_best_sol)
            except: global_best_cost = float('inf')
        
        total_time = time.time() - start_time
        if global_best_cost != float('inf'):
            print(f"\n[AnalyticalSolver] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Total Time: {total_time/60:.1f}m | Total Sims: {self.ctx.sim_count:,})")
        else:
            print(f"\n[AnalyticalSolver] EXECUTION ABORTED. No valid solutions.")
        
        real_diams = [self.ctx.diameters[i] for i in global_best_sol] if global_best_sol else []
        return real_diams