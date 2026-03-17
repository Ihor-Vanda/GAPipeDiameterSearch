import math
import random
import time
import bisect
import itertools
import multiprocessing
import networkx as nx
import numpy as np

# Виносимо імпорти на рівень модуля (Усунення прихованого багу продуктивності)
try:
    from .fast_math import fast_avg_hamming
    HAS_FAST_MATH = True
except ImportError:
    HAS_FAST_MATH = False

from .context import SolverContext
from .pool import SolutionPool
from .local_search import LocalSearch
from .kicks import KickStrategies

# =====================================================================
# КЛАС 1: ФАБРИКА ЗЕРЕН (SEED FACTORY)
# Відповідає за генерацію стартових рішень, засновану на фізиці EPANET
# =====================================================================
class SeedFactory:
    def __init__(self, ctx, local_search, n_workers, beam_width):
        self.ctx = ctx
        self.ls = local_search
        self.n_workers = n_workers
        self.BEAM_WIDTH = beam_width

    def calculate_ideal_d(self, flow, target_v):
        if abs(flow) < 1e-6: return 0
        d_ideal = math.sqrt((4.0 * abs(flow)) / (math.pi * target_v))
        pos = bisect.bisect_left(self.ctx.diameters, d_ideal)
        if pos < len(self.ctx.diameters): return pos
        return self.ctx.max_d_idx

    def make_reserve_pool(self, size=4):
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

    def make_diverse_seeds(self):
        seeds = []
        for v in [1.0, 1.2, 0.8]:
            idx_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            for _ in range(10):
                real_diams = [self.ctx.diameters[i] for i in idx_sol]
                flows, _ = self.ctx.simulator.get_hydraulic_state(real_diams)
                new_idx = [self.calculate_ideal_d(q, v) for q in flows] 
                if new_idx == idx_sol: break
                idx_sol = new_idx
                
            _, p, feas, _ = self.ctx.get_cached_stats(idx_sol)
            if not feas or p < self.ctx.simulator.config.h_min:
                idx_sol, is_healed, _ = self.ls.heal_network(idx_sol, set())
                if not is_healed: continue 
            squeezed_sol = self.ls.gradient_squeeze(idx_sol, quick_mode=True)
            seeds.append(squeezed_sol)
        return seeds

    def make_warm_seeds(self, archive, worker_id=0, failed_basins=None):
        failed_basins = failed_basins or set()
        if not archive: return self.make_diverse_seeds()
        
        archive_idx = worker_id % len(archive)
        base_sol = archive[archive_idx][1] 
        
        force_cold = (worker_id == self.n_workers - 1)
        if force_cold:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🏴‍☠️ FORCED ADVENTURER (Epoch Diversity)")
            return self.make_diverse_seeds()

        seeds = []
        if worker_id % 4 == 3:
            role = "ADVENTURER"
        else:
            adjusted_id = worker_id - (worker_id // 4)
            elite_roles = ["EXPLOITER", "RELINKER", "ARCHITECT", "EXPLORER"]
            role = elite_roles[adjusted_id % 4]

        if role == "EXPLOITER":
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🎯 EXPLOITER (Micro-mutations on Archive {archive_idx+1})")
            top_sols = [base_sol] 
            n_perturb = max(2, self.ctx.num_pipes // 10) 
            deltas = [-1, 1]
            for sol in top_sols:
                seeds.append(list(sol))
                for _ in range(self.BEAM_WIDTH - 1):
                    perturbed = list(sol)
                    for p_idx in random.sample(range(self.ctx.num_pipes), n_perturb):
                        perturbed[p_idx] = max(0, min(self.ctx.max_d_idx, perturbed[p_idx] + random.choice(deltas)))
                    seeds.append(perturbed)
                    
        elif role == "RELINKER":
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🧬 RELINKER (Path-Relinking)")
            seeds.append(list(base_sol))
            if len(archive) >= 2:
                target_sol = archive[(archive_idx + 1) % len(archive)][1]
                diff_indices = [i for i in range(self.ctx.num_pipes) if base_sol[i] != target_sol[i]]
                for _ in range(self.BEAM_WIDTH - 1):
                    child = list(base_sol)
                    cross_points = random.sample(diff_indices, min(len(diff_indices), max(2, len(diff_indices)//2)))
                    for idx in cross_points: child[idx] = target_sol[idx]
                    seeds.append(child)
            else:
                div_seeds = self.make_diverse_seeds()
                while len(seeds) < self.BEAM_WIDTH and div_seeds:
                    seeds.append(div_seeds.pop(0))
                
        elif role == "ARCHITECT":
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🏛️ ARCHITECT (Consensus Fixing)")
            seeds.append(list(base_sol))
            consensus = list(base_sol)
            if len(archive) >= 3:
                for i in range(self.ctx.num_pipes):
                    vals = [a[1][i] for a in archive[:3]]
                    if vals.count(vals[0]) == len(vals): consensus[i] = vals[0]
            for _ in range(self.BEAM_WIDTH - 1):
                child = list(consensus)
                for p_idx in random.sample(range(self.ctx.num_pipes), max(2, self.ctx.num_pipes // 15)):
                    child[p_idx] = max(0, min(self.ctx.max_d_idx, child[p_idx] + random.choice([-2, -1, 1, 2])))
                seeds.append(child)
                
        else:
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🔭 EXPLORER (Macro-mutations on Archive {archive_idx+1})")
            top_sols = [base_sol]
            n_perturb = max(5, self.ctx.num_pipes // 5)
            deltas = [-2, -1, 1, 2]
            for sol in top_sols:
                seeds.append(list(sol))
                for _ in range(self.BEAM_WIDTH - 1):
                    perturbed = list(sol)
                    for p_idx in random.sample(range(self.ctx.num_pipes), n_perturb):
                        perturbed[p_idx] = max(0, min(self.ctx.max_d_idx, perturbed[p_idx] + random.choice(deltas)))
                    seeds.append(perturbed)
                    
        return seeds[:self.BEAM_WIDTH]

# =====================================================================
# КЛАС 2: АГЕНТ ПОШУКУ (ISLAND WORKER)
# =====================================================================
class IslandWorker:
    def __init__(self, ctx, kicker, local_search, worker_id, n_workers, max_sims, beam_width, network_class, global_archive, epoch):
        self.ctx = ctx
        self.kicker = kicker
        self.ls = local_search
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.max_sims = max_sims
        self.BEAM_WIDTH = beam_width
        self.network_class = network_class
        self.global_archive = global_archive
        self.epoch = epoch
        
        self.pool = SolutionPool(self.ctx)
        self.seeder = SeedFactory(self.ctx, self.ls, self.n_workers, self.BEAM_WIDTH)
        
        self.strategies = ["TOPO-DIV", "LOOP_BALANCE", "SYNC_TRIM", "FINISHER", "BOTTLENECK", 
                           "ZERO_SUM", "SUBMARINE", "SHOCK", "RUIN_RECREATE", "VNS_KICK", "ILS_PERTURBATION", "DIAM_DIVERSITY"]
        
        if nx.is_tree(self.ctx.base_G_flow) and "LOOP_BALANCE" in self.strategies:
            self.strategies.remove("LOOP_BALANCE")
            
        all_tracked = self.strategies + ["SEGMENT_RESTART", "IPC_CROSSOVER", "CORRIDOR_SEARCH"]
        self.strat_wins = {s: 1.0 for s in all_tracked}
        self.strat_tries = {s: 1.0 for s in all_tracked}
        
        n = self.ctx.num_pipes
        single_cands = {"SMALL": max(6, n//5), "MEDIUM": max(6, n//10), 
                        "LARGE": max(6, n//20), "XLARGE": 10}[self.network_class]
        self.SINGLE_CANDIDATES = single_cands
        
        self.stagnation_counter = 0
        self.run_best_cost = float('inf')
        self.run_best_sol = None
        self.global_best_cost = float('inf')
        self.last_published_cost = float('inf')
        self.progress_ratio = 0.0
        self.is_late_game = False
        
        self.last_injected_peer = {}
        self.rescue_fired = False
        self.corridor_pool_streak = 0
        self.loop_balance_failed_pipes = {}
        self._last_global_improvement_sim = 0

    def run(self, time_budget, global_best_cost, shared_progress):
        self.global_best_cost = global_best_cost
        start_time = time.time()
        self.pool.clear_all()
        
        seeds = self.seeder.make_diverse_seeds() if self.epoch == 0 else self.seeder.make_warm_seeds(self.global_archive, self.worker_id)
        self.reserve_pool = self.seeder.make_reserve_pool(size=4)
        
        best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds]) if seeds else global_best_cost
        self.base_dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001 
        
        self._initialize_seeds(seeds)
        
        self.stag_limit = 4 
        round_idx = 0
        epoch_start_sims = self.ctx.sim_count 
        self._last_global_improvement_sim = self.ctx.sim_count
        just_flushed = False
        
        while True:
            self.pool.current_round = round_idx 
            self.base_dyn_bonus = min(self.run_best_cost, self.global_best_cost) * 0.001 * random.uniform(0.70, 1.30)
            
            # --- 1. IPC та Порятунок ---
            gb = shared_progress.get('global_best') if shared_progress else None
            if shared_progress is not None:
                self._process_ipc(shared_progress, gb)
                self._check_rescue(shared_progress, gb)
                shared_progress[self.worker_id] = {"round": round_idx + 1, "sims": self.ctx.sim_count, "best_cost": self.run_best_cost}
            
            # --- 2. Таймінги ---
            elapsed = time.time() - start_time
            epoch_sims = self.ctx.sim_count - epoch_start_sims
            if elapsed > time_budget or epoch_sims >= self.max_sims: break
                
            just_flushed = self.rescue_fired
            self.rescue_fired = False
            
            self.progress_ratio = min(1.0, epoch_sims / max(1, self.max_sims)) 
            self.is_late_game = self.progress_ratio > 0.5 
            self.stag_limit = 4 + int(4 * self.progress_ratio) 
            
            # --- 3. Перевірка стагнації ---
            self._check_mini_restart(shared_progress, gb)
            if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()
            
            # --- 4. Мікро-полірування (SWAP) ---
            self._apply_swap(round_idx, shared_progress)

            # --- 5. Макро-Кіки (Стратегії) ---
            if self.stagnation_counter >= 1 or round_idx % 2 == 0:
                self._apply_kick(round_idx, shared_progress, gb)

            # --- 6. Генерація мутацій пулу ---
            next_gen = self._generate_mutations()
            
            # --- 7. Градієнтний спуск та Beam Search ---
            if next_gen:
                next_gen.sort(key=lambda x: x[0])
                just_flushed = self._beam_search_and_update(next_gen, just_flushed, shared_progress, round_idx)
            else:
                self._emergency_respawn()
                just_flushed = True

            # --- 8. Забування історії UCB1 ---
            if round_idx > 0 and round_idx % 10 == 0:
                for s in self.strat_wins:
                    self.strat_wins[s] *= 0.80
                    self.strat_tries[s] = max(1.0, self.strat_tries[s] * 0.80)

            round_idx += 1
                
        if shared_progress is not None:
            shared_progress[self.worker_id] = {"round": "DONE", "sims": self.ctx.sim_count, "best_cost": self.run_best_cost}

        return self.run_best_cost, self.run_best_sol

    # ================= ПРИВАТНІ МЕТОДИ ВОКЕРА =================

    def _initialize_seeds(self, seeds):
        valid_sols = []
        for s in seeds:
            c, p, feas, _ = self.ctx.get_cached_stats(s)
            p_surplus = p - self.ctx.simulator.config.h_min
            score = c - (p_surplus * self.base_dyn_bonus)
            self.pool.active_pool.append((score, c, s))
            self.pool.add_to_tabu(s, c)
            if feas and p >= self.ctx.simulator.config.h_min:
                valid_sols.append((c, s))
                
        if valid_sols:
            self.run_best_cost = min(x[0] for x in valid_sols)
            self.run_best_sol = next(x[1] for x in valid_sols if x[0] == self.run_best_cost)
        else:
            best_seed = min(self.pool.active_pool, key=lambda x: x[1])
            self.run_best_cost = best_seed[1] 
            self.run_best_sol = best_seed[2]
            
        self.ctx.log(f"   > Seeds initialized. Baseline Target: {self.run_best_cost/1e6:.4f}M$")

    def _process_ipc(self, shared_progress, gb):
        if gb and gb[0] < self.global_best_cost:
            self.global_best_cost = gb[0]
            self._last_global_improvement_sim = self.ctx.sim_count
            self.ctx.log(f"   > [IPC] 📡 Received new Global Bound from peer: {self.global_best_cost/1e6:.4f}M$")
        
        for i in range(self.n_workers):
            if i == self.worker_id: continue
            peer = shared_progress.get(f'best_sol_{i}')
            last_cost = self.last_injected_peer.get(i, float('inf'))
            
            if peer and peer[0] < self.run_best_cost * 0.995 and peer[0] < last_cost - 1.0:
                is_adventurer = (self.n_workers >= 4 and self.worker_id == self.n_workers - 1)
                if is_adventurer and self.progress_ratio < 0.85:
                    continue 
                    
                peer_sol = list(peer[1])
                c_p, p_p, feas_p, _ = self.ctx.get_cached_stats(peer_sol)
                if feas_p and p_p >= self.ctx.simulator.config.h_min and not self.pool.is_basin_tabu(peer_sol):
                    p_surplus = p_p - self.ctx.simulator.config.h_min
                    score = c_p - (p_surplus * self.base_dyn_bonus)
                    self.pool.active_pool.append((score, c_p, peer_sol))
                    self.ctx.log(f"     [IPC] 💉 Passively injected peer W{i+1} solution ({c_p/1e6:.4f}M$)")
                    self.last_injected_peer[i] = peer[0] 

    def _check_rescue(self, shared_progress, gb):
        global_lag = (self.run_best_cost - self.global_best_cost) / max(self.global_best_cost, 1)
        sims_since_global_update = self.ctx.sim_count - self._last_global_improvement_sim
        no_global_progress = sims_since_global_update > self.max_sims * 0.15 
        
        if (global_lag > 0.02 or no_global_progress) and self.stagnation_counter >= self.stag_limit * 2:
            if no_global_progress and gb and gb[1]:
                self.ctx.log(f"   [RESCUE] No global progress for {sims_since_global_update//1000}k sims. Forcing basin escape!")
            elif global_lag > 0.02 and gb and gb[1]:
                self.ctx.log(f"   [RESCUE] Worker lagging by {global_lag:.1%}. Abandoning dead basin and adopting Global Best {gb[0]/1e6:.4f}M$!")
                
            if gb and gb[1]:
                gb_sol = list(gb[1])
                c_gb, p_gb, feas_gb, _ = self.ctx.get_cached_stats(gb_sol)
                if feas_gb and p_gb >= self.ctx.simulator.config.h_min:
                    self.run_best_cost = c_gb
                    self.run_best_sol = gb_sol
                    p_surplus = p_gb - self.ctx.simulator.config.h_min
                    score = c_gb - (p_surplus * self.base_dyn_bonus)
                    self.pool.active_pool.insert(0, (score - 1e6, c_gb, gb_sol))
                    self.stagnation_counter = 0
                    self.pool.kick_tabu_set.clear()
                    self.rescue_fired = True 

    def _check_mini_restart(self, shared_progress, gb):
        base_mult = 6 if self.ctx.num_pipes >= 200 else 10
        effective_mult = base_mult if not self.is_late_game else max(4, base_mult // 2)
        
        if self.stagnation_counter >= self.stag_limit * effective_mult:
            self.ctx.log(f"   [MINI-RESTART] {self.stagnation_counter} rounds without progress. Soft epoch restart.")
            self.pool.tabu_fingerprints.clear()
            self.pool.kick_tabu_set.clear()
            self.pool.basin_tabu.clear()
            self.pool.active_pool.clear()
            
            if gb and gb[1]:
                archive_sol = list(gb[1])
                new_seeds = []
                n_perturb = max(10, self.ctx.num_pipes // 3) if self.is_late_game else max(8, self.ctx.num_pipes // 4)
                
                for _ in range(4): 
                    perturbed = list(archive_sol)
                    for p_idx in random.sample(range(self.ctx.num_pipes), n_perturb):
                        perturbed[p_idx] = max(0, min(self.ctx.max_d_idx, perturbed[p_idx] + random.choice([-2, -1, 1, 2])))
                        
                    healed, ok, _ = self.ls.heal_network(perturbed, set())
                    if ok:
                        squeezed = self.ls.gradient_squeeze(healed, max_passes=2, quick_mode=True, dyn_bonus=self.base_dyn_bonus)
                        if not self.pool.is_basin_tabu(squeezed):
                            new_seeds.append(squeezed)
                            
                if new_seeds:
                    for seed in new_seeds:
                        c, p, feas, _ = self.ctx.get_cached_stats(seed)
                        if feas and p >= self.ctx.simulator.config.h_min:
                            score = c - ((p - self.ctx.simulator.config.h_min) * self.base_dyn_bonus)
                            self.pool.active_pool.append((score, c, seed))
                    self.stagnation_counter = 0
                    self.ctx.log(f"   [MINI-RESTART] Injected {len(new_seeds)} fresh seeds. Tabu cleared.")

    def _apply_swap(self, round_idx, shared_progress):
        if round_idx > 0 and round_idx % 8 == 0:
            swapped = self.ls.swap_search(self.run_best_sol, self.base_dyn_bonus)
            c, p, feas, _ = self.ctx.get_cached_stats(swapped)
            if feas and p >= self.ctx.simulator.config.h_min and c < self.run_best_cost:
                is_ghost = False
                if (self.run_best_cost - c) > (self.run_best_cost * 0.02):
                    is_ghost = self.ctx.is_ghost_solution(swapped, c)
                    
                if is_ghost:
                    self.ctx.log(f"   > [SHIELD] Swap illusion blocked ({c/1e6:.4f}M$)!")
                else:
                    diff = self.run_best_cost - c
                    self.run_best_cost, self.run_best_sol = c, swapped
                    self.ctx.log(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                    self._update_global_best(shared_progress)
                    
                    self.stagnation_counter = max(0, self.stagnation_counter - 1)
                    self.pool.kick_tabu_set.clear()
                    
                    p_surplus = p - self.ctx.simulator.config.h_min
                    eff_bonus = self.base_dyn_bonus * 0.3 if p_surplus < 1.0 else (self.base_dyn_bonus * 2.0 if p_surplus > 15.0 else self.base_dyn_bonus)
                    self.pool.active_pool.insert(0, (c - (p_surplus * eff_bonus) - (self.run_best_cost * 0.1), c, swapped))

    def _apply_kick(self, round_idx, shared_progress, gb):
        n_total = sum(self.strat_tries.values())
        peer_archive = []
        if shared_progress is not None:
            for i in range(self.n_workers):
                if i != self.worker_id:
                    peer_data = shared_progress.get(f'best_sol_{i}')
                    if peer_data: peer_archive.append(peer_data)

        peer_to_cross = None
        if self.stagnation_counter >= self.stag_limit * 3:
            strategy = "SEGMENT_RESTART"
            self.pool.add_basin_to_tabu(self.run_best_sol) 
        elif self.stagnation_counter >= self.stag_limit * 2 and peer_archive:
            corridor_peers = [p for p in peer_archive if 30 <= self.pool.hamming_distance(self.run_best_sol, p[1]) <= 70]
            diverse_peers = [p for p in peer_archive if self.pool.hamming_distance(self.run_best_sol, p[1]) > 70]
            if corridor_peers:
                strategy = "CORRIDOR_SEARCH"
                peer_to_cross = random.choice(corridor_peers)
            elif diverse_peers:
                strategy = "IPC_CROSSOVER"
                peer_to_cross = min(diverse_peers, key=lambda x: x[0])
            else:
                strategy = "ILS_PERTURBATION"
        elif self.stagnation_counter >= self.stag_limit * 1.5:
            strategy = "ILS_PERTURBATION"
        elif self.stagnation_counter > 0 and self.stagnation_counter % 12 == 0:
            strategy = "RUIN_RECREATE"
        elif self.stagnation_counter > 0 and self.stagnation_counter % 7 == 0:
            strategy = "VNS_KICK"
        else:
            valid_strats = [s for s in self.strategies if s in self.strat_wins]
            if not valid_strats:
                strategy = "ILS_PERTURBATION"
            else:
                strategy = max(valid_strats, key=lambda s: (self.strat_wins[s] / self.strat_tries[s]) + math.sqrt(2 * math.log(max(1, n_total)) / self.strat_tries[s]))
            
        self.strat_tries[strategy] += 1
        self.ctx.log(f"[FORCE] Escalation Level: {self.stagnation_counter}/{self.stag_limit} -> Applying '{strategy}'...")
        
        if self.stagnation_counter >= self.stag_limit and self.pool.active_pool:
            valid_pool = [x for x in self.pool.active_pool if self.ctx.get_cached_stats(x[2])[1] >= self.ctx.simulator.config.h_min]
            source_pool = valid_pool if valid_pool else self.pool.active_pool
            kick_target = max([x[2] for x in source_pool], key=lambda s: self.pool.hamming_distance(s, self.run_best_sol))
        else:
            kick_mode = round_idx % 3
            if kick_mode == 0: kick_target = self.run_best_sol
            elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
            else:
                pool_sols = [x[2] for x in self.pool.active_pool[:3]]
                kick_target = random.choice(pool_sols) if pool_sols else self.run_best_sol
        
        forced_sol, locked, path_sig, failed_pipe_id = None, None, None, -1
        
        if strategy == "SEGMENT_RESTART": forced_sol, locked, log_msg = self.kicker.segment_restart_kick(kick_target, self.base_dyn_bonus)
        elif strategy == "IPC_CROSSOVER": forced_sol, locked, log_msg = self.kicker.crossover_with_peer_kick(kick_target, peer_to_cross[1], self.run_best_cost, peer_to_cross[0])
        elif strategy == "CORRIDOR_SEARCH": forced_sol, locked, log_msg = self.kicker.corridor_search_kick(kick_target, peer_to_cross[1])
        elif strategy == "ILS_PERTURBATION": forced_sol, locked, log_msg = self.kicker.ils_perturbation_kick(kick_target, self.stagnation_counter)
        elif strategy == "VNS_KICK": forced_sol, locked, log_msg = self.kicker.vns_structured_kick(kick_target, self.stagnation_counter)
        elif strategy == "SHOCK": forced_sol, locked, log_msg = self.kicker.forcing_hand_kick(kick_target)
        elif strategy == "BOTTLENECK": forced_sol, locked, log_msg = self.kicker.upstream_bottleneck_kick(kick_target)
        elif strategy == "LOOP_BALANCE": 
            forced_sol, locked, log_msg, failed_pipe_id = self.kicker.loop_balancing_kick(kick_target, self.base_dyn_bonus, self.loop_balance_failed_pipes, round_idx)
        elif strategy == "DIAM_DIVERSITY": forced_sol, locked, log_msg = self.kicker.diameter_diversity_kick(kick_target, self.stagnation_counter) 
        elif strategy == "RUIN_RECREATE": forced_sol, locked, log_msg = self.kicker.ruin_and_recreate_kick(kick_target, self.stagnation_counter) 
        elif strategy == "FINISHER": forced_sol, locked, log_msg = self.kicker.micro_trim_kick(kick_target)
        elif strategy == "SYNC_TRIM": forced_sol, locked, log_msg = self.kicker.sync_trim_kick(kick_target, self.base_dyn_bonus)
        elif strategy == "ZERO_SUM": forced_sol, locked, log_msg = self.kicker.zero_sum_shift_kick(kick_target)
        elif strategy == "SUBMARINE": forced_sol, locked, log_msg = self.kicker.submarine_oscillation_kick(kick_target)
        else: forced_sol, locked, log_msg, path_sig = self.kicker.topological_inversion_kick(kick_target, self.pool.kick_tabu_set)

        if failed_pipe_id != -1: self.loop_balance_failed_pipes[failed_pipe_id] = round_idx
        if not forced_sol or not locked: return

        self.ctx.log(f"     -> {log_msg}")
        
        stagnation_relax = min(0.05, (self.stagnation_counter / max(1, self.stag_limit)) * 0.015)
        base_margin = 0.08 if self.ctx.num_pipes >= 200 else 0.05
        if self.stagnation_counter >= self.stag_limit * 2.5: base_margin *= 1.5 
            
        water_level = max(self.global_best_cost * (1.0 + base_margin * (1.0 - self.progress_ratio) + stagnation_relax), self.run_best_cost * 1.03)

        if strategy == "SEGMENT_RESTART":
            final_sol = forced_sol 
            self.pool.basin_tabu.clear() 
        else:
            is_heavy = strategy in ["ILS_PERTURBATION", "VNS_KICK", "RUIN_RECREATE", "IPC_CROSSOVER", "CORRIDOR_SEARCH", "SUBMARINE"]
            base_quick = 2 if self.ctx.num_pipes >= 200 else 1
            quick_passes = base_quick if is_heavy else base_quick + 1
            locked_for_squeeze = set() if strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "CORRIDOR_SEARCH"] else locked
            
            quick_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked_for_squeeze, max_passes=quick_passes, quick_mode=True, dyn_bonus=self.base_dyn_bonus)
            quick_cost, quick_p, quick_f, _ = self.ctx.get_cached_stats(quick_sol)
            
            tolerance = 1.15 if self.stagnation_counter > self.stag_limit * 2 else (1.05 if is_heavy else 1.02)
            hyperband_threshold = max(self.run_best_cost * tolerance, water_level * 1.02)
            
            if quick_f and quick_p >= self.ctx.simulator.config.h_min and (quick_cost < hyperband_threshold):
                if self.ctx.num_pipes >= 200 and self.progress_ratio > 0.7: deep_passes = 6 
                elif is_heavy: deep_passes = 5
                else: deep_passes = 3 if self.progress_ratio > 0.5 else 6
                self.ctx.log(f"        [HYPERBAND] Promising path detected ({quick_cost/1e6:.2f}M$). Deep Squeeze ({deep_passes} passes)...")
                final_sol = self.ls.gradient_squeeze(quick_sol, locked_pipes=locked_for_squeeze, max_passes=deep_passes, dyn_bonus=self.base_dyn_bonus)
            else:
                final_sol = quick_sol
        
        c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
        if feas and c > self.run_best_cost * 1.5:
            self.ctx.log(f"     -> Hard-Rejected (Cost Explosion): {c/1e6:.4f}M$")
            feas = False
        
        if feas and p >= self.ctx.simulator.config.h_min:
            p_surplus = p - self.ctx.simulator.config.h_min
            eff_bonus = self.base_dyn_bonus * 0.3 if p_surplus < 1.0 else (self.base_dyn_bonus * 2.0 if p_surplus > 15.0 else self.base_dyn_bonus)
            
            if p_surplus < 2.0:
                final_sol = self.ls.gradient_squeeze(final_sol, locked_pipes=set(), max_passes=3, quick_mode=True, dyn_bonus=eff_bonus)
                c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                p_surplus = p - self.ctx.simulator.config.h_min if feas else p_surplus
                eff_bonus = self.base_dyn_bonus * 0.3 if p_surplus < 1.0 else self.base_dyn_bonus
            
            score = c - (p_surplus * eff_bonus)
            
            if c < self.run_best_cost:
                is_ghost = False
                if (self.run_best_cost - c) > (self.run_best_cost * 0.02):
                    is_ghost = self.ctx.is_ghost_solution(final_sol, c)
                    
                if is_ghost:
                    self.ctx.log(f"   > [SHIELD] Force illusion blocked ({c/1e6:.4f}M$)!")
                else:
                    diff = self.run_best_cost - c
                    self.run_best_cost, self.run_best_sol = c, final_sol
                    self.pool.active_pool.insert(0, (score, c, final_sol))
                    
                    if diff > (self.run_best_cost * 0.001):
                        self.stagnation_counter = 0 
                    else:
                        self.stagnation_counter = max(0, self.stagnation_counter - 2)
                    
                    improvement_pct = diff / self.run_best_cost
                    reward = 5.0 if improvement_pct > 0.01 else (3.0 if improvement_pct > 0.001 else 1.0)
                    
                    self.strat_wins[strategy] += reward
                    self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                    self._update_global_best(shared_progress)

            elif (c < water_level and not self.pool.is_basin_tabu(final_sol)) or strategy == "SEGMENT_RESTART":
                pool_gap = (c - self.run_best_cost) / max(self.run_best_cost, 1)
                max_pool_gap = 0.06 if self.is_late_game else 0.12
                
                if self.ctx.num_pipes >= 200 and pool_gap > max_pool_gap and strategy != "SEGMENT_RESTART":
                    self.ctx.log(f"     -> Pool Filtered (gap {pool_gap:.1%}): {c/1e6:.4f}M$")
                else:
                    self.pool.active_pool.append((score * 1.05, c, final_sol))
                    if strategy == "CORRIDOR_SEARCH":
                        self.corridor_pool_streak += 1
                        if self.corridor_pool_streak <= 3: self.stagnation_counter = 0 
                    elif strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "SEGMENT_RESTART", "RUIN_RECREATE", "VNS_KICK", "ZERO_SUM", "SUBMARINE"]:
                        self.stagnation_counter = 0 
                        self.corridor_pool_streak = 0
                    self.strat_wins[strategy] += 0.5 
                    self.ctx.log(f"     -> Added to Pool (Water Level Accept): {c/1e6:.4f}M$")
                    
            elif self.stagnation_counter >= self.stag_limit * 3.5 and not self.pool.is_basin_tabu(final_sol):
                self.pool.active_pool.append((score * 1.20, c, final_sol))
                self.stagnation_counter = int(self.stag_limit * 0.5) 
                self.ctx.log(f"     -> 🚀 HAIL MARY ACCEPT (Forced Escape): {c/1e6:.4f}M$")
            else:
                self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {c/1e6:.4f}M$")
                if strategy == "SEGMENT_RESTART": self.stagnation_counter = int(self.stag_limit * 1.5)
            if path_sig: self.pool.kick_tabu_set.add(path_sig)
        else:
             self.ctx.log(f"     -> Injection/Squeeze Failed: Infeasible/Exploded")

    def _generate_mutations(self):
        next_gen = []
        for _, _, parent_sol in self.pool.active_pool:
            unit_losses = self.ctx.get_cached_heuristics(parent_sol)
            high_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i], reverse=True)
            low_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i]) 
            
            _, p, feas, _ = self.ctx.get_cached_stats(parent_sol)
            parent_p_surplus = (p - self.ctx.simulator.config.h_min) if feas else 0.0
            
            if parent_p_surplus > 10.0: downgrade_limit, upgrade_limit = 15, self.SINGLE_CANDIDATES
            elif parent_p_surplus < 2.0: downgrade_limit, upgrade_limit = 3, self.SINGLE_CANDIDATES // 2
            else: downgrade_limit, upgrade_limit = 8, self.SINGLE_CANDIDATES
            
            if self.network_class in ("LARGE", "XLARGE"):
                top_k = max(20, self.ctx.num_pipes // 5)
                focus_pipes = set(self.ls.get_high_impact_pipes(parent_sol, top_k))
                high_friction = [pi for pi in high_friction if pi in focus_pipes]
                low_friction = [pi for pi in low_friction if pi in focus_pipes]
            
            for pipe_idx in high_friction[:upgrade_limit]:
                s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "upgrade", self.base_dyn_bonus)
                if sol: next_gen.append((s, c, sol))
            for pipe_idx in low_friction[:downgrade_limit]:
                 s, c, sol = self.ls.evaluate_candidate(parent_sol, [pipe_idx], "downgrade", self.base_dyn_bonus)
                 if sol: next_gen.append((s, c, sol))
                 
            if len(high_friction) >= 2:
                combo_limit = 5 if parent_p_surplus >= 2.0 else 3
                for p1, p2 in itertools.combinations(high_friction[:combo_limit], 2): 
                    s, c, sol = self.ls.evaluate_candidate(parent_sol, [p1, p2], "upgrade", self.base_dyn_bonus)
                    if sol: next_gen.append((s, c, sol))
        return next_gen

    def _beam_search_and_update(self, next_gen, just_flushed, shared_progress, round_idx):
        unique_next_pool = []
        found_new_record = False
        max_d = max(2, self.ctx.num_pipes // 7) 
        min_dist = max(1, int(max_d * (1.0 - (self.progress_ratio ** 2))))
        
        for rank, (score, cost, sol) in enumerate(next_gen):
            if self.pool.is_tabu(sol, cost): continue
            if len(unique_next_pool) < self.BEAM_WIDTH:
                if rank == 0: 
                    refined_sol = self.ls.gradient_squeeze(sol, max_passes=12, dyn_bonus=self.base_dyn_bonus)
                elif rank < 3: 
                    refined_sol = self.ls.gradient_squeeze(sol, max_passes=5, quick_mode=not self.is_late_game, dyn_bonus=self.base_dyn_bonus)
                else: 
                    refined_sol = sol 

                is_diverse = True
                for _, _, peer_sol in unique_next_pool:
                    if self.pool.hamming_distance(refined_sol, peer_sol) < min_dist:
                        is_diverse = False; break
                        
                if not is_diverse and not just_flushed: continue

                real_cost, p_min, feas_min, _ = self.ctx.get_cached_stats(refined_sol)
                if feas_min:
                    real_p_surplus = p_min - self.ctx.simulator.config.h_min
                    is_strictly_valid = (real_p_surplus >= 0.0)
                    is_epsilon_valid = (real_p_surplus >= -0.5) 
                    
                    if is_strictly_valid or is_epsilon_valid:
                        if is_strictly_valid:
                            eff_bonus = self.base_dyn_bonus * 0.3 if real_p_surplus < 1.0 else (self.base_dyn_bonus * 2.0 if real_p_surplus > 15.0 else self.base_dyn_bonus)
                            score = real_cost - (real_p_surplus * eff_bonus)
                        else:
                            score = real_cost + (abs(real_p_surplus) * self.base_dyn_bonus * 50.0)
                        
                        unique_next_pool.append((score, real_cost, refined_sol))
                        self.pool.add_to_tabu(refined_sol, real_cost)
                        
                        if is_strictly_valid and real_cost < self.run_best_cost:
                            is_ghost = False
                            if (self.run_best_cost - real_cost) > (self.run_best_cost * 0.02):
                                is_ghost = self.ctx.is_ghost_solution(refined_sol, real_cost)
                                
                            if is_ghost:
                                self.ctx.log(f"   > [SHIELD] Beam illusion blocked ({real_cost/1e6:.4f}M$).")
                                continue 

                            diff = self.run_best_cost - real_cost
                            self.run_best_cost, self.run_best_sol = real_cost, refined_sol
                            
                            if diff > (self.run_best_cost * 0.001):
                                found_new_record = True 
                                self.ctx.log(f"   > [R{round_idx+1}] 💎 New RECORD: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                            else:
                                self.ctx.log(f"   > [R{round_idx+1}] 💎 Micro-Step: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                            
                            self._update_global_best(shared_progress)

        dynamic_beam = max(3, int(self.BEAM_WIDTH * (1.0 + 0.5 * (1.0 - self.progress_ratio))))
        self.pool.active_pool = unique_next_pool[:dynamic_beam]
        
        if found_new_record or just_flushed:
            self.stagnation_counter = 0
            self.pool.kick_tabu_set.clear()
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter % 8 == 0: self.pool.kick_tabu_set.clear()
            
        self._emergency_pool_diversity()
        return found_new_record or just_flushed

    def _emergency_pool_diversity(self):
        if len(self.pool.active_pool) >= 3:
            sols = [x[2] for x in self.pool.active_pool]
            if HAS_FAST_MATH:
                pool_matrix = np.array(sols, dtype=np.int32)
                avg_dist = fast_avg_hamming(pool_matrix)
            else:
                pairs = list(itertools.combinations(range(len(sols)), 2))
                avg_dist = sum(self.pool.hamming_distance(sols[a], sols[b]) for a, b in pairs) / len(pairs)
            
            base_div = self.ctx.num_pipes // 8
            diversity_threshold = max(1, int(base_div * (1.0 - self.progress_ratio)))
            
            if avg_dist < diversity_threshold and self.reserve_pool:
                fresh = self.reserve_pool.pop(0) 
                c, p, feas, _ = self.ctx.get_cached_stats(fresh)
                if feas:
                    p_surplus = p - self.ctx.simulator.config.h_min
                    score = c - (p_surplus * self.base_dyn_bonus)
                    self.pool.active_pool[-1] = (score, c, fresh) 

    def _emergency_respawn(self):
        self.ctx.log("     [EMERGENCY] Beam search deadlocked (no valid/diverse children). Flushing pool & tabu!")
        self.pool.active_pool.clear()
        self.pool.tabu_fingerprints.clear()
        self.stagnation_counter += 1
        
        forced_rescue, _, _ = self.kicker.ils_perturbation_kick(self.run_best_sol, 15)
        if forced_rescue is not None:
            c, p, feas, _ = self.ctx.get_cached_stats(forced_rescue)
            if feas and p >= self.ctx.simulator.config.h_min:
                self.pool.active_pool.append((c - 1e6, c, forced_rescue))
                self.stagnation_counter = 0 
        
        if not self.pool.active_pool:
            healed_seed, _, _ = self.kicker.vns_structured_kick(self.run_best_sol, stagnation_level=8)
            if healed_seed is not None:
                c, p, feas, _ = self.ctx.get_cached_stats(healed_seed)
                if feas:
                    self.pool.active_pool.append((c - 1e6, c, healed_seed))
                    self.stagnation_counter = 0
                    self.pool.kick_tabu_set.clear()
            else:
                safe_seed = list(self.run_best_sol)
                p_idx = random.randint(0, self.ctx.num_pipes - 1)
                safe_seed[p_idx] = min(self.ctx.max_d_idx, safe_seed[p_idx] + 2)
                h_seed, ok, _ = self.ls.heal_network(safe_seed, {p_idx})
                if ok:
                    c, _, _, _ = self.ctx.get_cached_stats(h_seed)
                    self.pool.active_pool.append((c - 1e6, c, h_seed))
                    self.stagnation_counter = 0

    def _update_global_best(self, shared_progress):
        if self.run_best_cost < self.last_published_cost:
            if shared_progress is not None:
                shared_progress[f'best_sol_{self.worker_id}'] = (self.run_best_cost, list(self.run_best_sol))
            self.last_published_cost = self.run_best_cost

        if self.run_best_cost < self.global_best_cost:
            self.global_best_cost = self.run_best_cost
            self._last_global_improvement_sim = self.ctx.sim_count
            if shared_progress is not None:
                current_gb = shared_progress.get('global_best', (float('inf'), []))
                if self.run_best_cost < current_gb[0]:
                    shared_progress['global_best'] = (self.run_best_cost, list(self.run_best_sol))


# =====================================================================
# КЛАС 3: ОРКЕСТРАТОР КЛАСТЕРА (ANALYTICAL SOLVER)
# =====================================================================
class AnalyticalSolver:
    @classmethod
    def worker_task(cls, args):
        try:
            diams            = args[0]
            v_opt            = args[1]
            time_budget      = args[2]
            global_best_cost = args[3]
            global_archive   = args[4]
            seed_mod         = args[5]
            worker_id        = args[6]
            shared_progress  = args[7]
            log_dir          = args[8]
            epoch            = args[9]
            failed_basins    = args[10]
            max_sims         = args[11] if len(args) > 11 else float('inf')
            n_workers        = args[12] if len(args) > 12 else (len(shared_progress) if shared_progress else 1)
            
            import sys
            import __main__
            sim = getattr(__main__, 'worker_sim_instance', None)
            if sim is None: return (float('inf'), None, None, 0, set())

            ctx = SolverContext(sim, diams, v_opt=v_opt)
            
            import os
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                ctx.log_file = os.path.join(log_dir, f"epoch_{epoch+1}_worker_{worker_id+1:02d}.txt")
            else:
                ctx.log_file = None 
                
            ls = LocalSearch(ctx)
            kicker = KickStrategies(ctx, ls)
            
            random.seed(seed_mod)
            np.random.seed(seed_mod)
            
            n = ctx.num_pipes
            network_class = "SMALL" if n < 50 else ("MEDIUM" if n < 200 else ("LARGE" if n < 1000 else "XLARGE"))
            beam_width = 8 if network_class in ["LARGE", "XLARGE"] else 5
            
            worker = IslandWorker(ctx, kicker, ls, worker_id, n_workers, max_sims, beam_width, network_class, global_archive, epoch)
            c_best, sol_best = worker.run(time_budget, global_best_cost, shared_progress)
            
            return c_best, sol_best, None, ctx.sim_count, worker.pool.basin_tabu
        except Exception as e:
            print(f"     [CRITICAL WORKER ERROR] Worker {args[6] + 1} failed: {e}")
            return float('inf'), None, None, 0, set()

    def __init__(self, simulator_instance, available_diameters, v_opt=1.0, max_sims=None, time_limit_sec=None, pool=None, log_dir=None, n_workers=1):
        self.ctx = SolverContext(simulator_instance, available_diameters, v_opt)
        self.ls = LocalSearch(self.ctx)
        self.mp_pool = pool
        self.log_dir = log_dir
        self.n_workers = n_workers
        
        n = self.ctx.num_pipes
        if n < 50: self.network_class = "SMALL"
        elif n < 200: self.network_class = "MEDIUM"
        elif n < 1000: self.network_class = "LARGE"
        else: self.network_class = "XLARGE"
        
        self.BEAM_WIDTH = 8 if self.network_class in ["LARGE", "XLARGE"] else 5
        self.seeder = SeedFactory(self.ctx, self.ls, self.n_workers, self.BEAM_WIDTH)
        self.pool = SolutionPool(self.ctx) 
            
        if max_sims is not None:
            self.max_sims = max_sims
            self.time_limit_sec = float('inf')
        else:
            self.max_sims = float('inf')
            self.time_limit_sec = time_limit_sec or 3600.0

    def solve_standalone(self, max_sims=None, time_limit_sec=None):
        print("\n[AnalyticalSolver] ⚡ Initiating Island Model Search...\n")
        start_time = time.time()
        global_best_cost = float('inf')
        global_best_sol = None
        global_archive = []

        epochs = {"SMALL": 4, "MEDIUM": 4, "LARGE": 5, "XLARGE": 5}[self.network_class]
        time_per_epoch = self.time_limit_sec / epochs

        manager = multiprocessing.Manager() if self.mp_pool else None
        shared_progress = manager.dict() if manager else None
        if shared_progress:
            for i in range(self.n_workers): shared_progress[i] = 0

        cumulative_epoch_sims = 0 
        global_failed_basins = set()

        for epoch in range(epochs):
            mode_str = "PARALLEL" if self.mp_pool else "SEQUENTIAL"
            print("="*46)
            print(f" [EPOCH {epoch+1}/{epochs}] {mode_str} Workers: {self.n_workers} | Time Limit: {time_per_epoch/60:.1f} min")
            print("="*46)
            
            seed_modifier = random.randint(1, 10000)
            tasks = []
            for i in range(self.n_workers):
                tasks.append((
                    self.ctx.diameters, self.ctx.v_opt, time_per_epoch, 
                    global_best_cost, global_archive, 
                    seed_modifier + i, i, shared_progress, self.log_dir, epoch,
                    global_failed_basins, self.max_sims, self.n_workers
                ))

            epoch_results = []
            
            if self.mp_pool:
                async_results = []
                for t in tasks:
                    res = self.mp_pool.apply_async(self.worker_task, (t,))
                    async_results.append((t[6], res))

                epoch_start_time = time.time()
                last_print_time = 0
                print_interval = {"SMALL": 1.0, "MEDIUM": 15.0, "LARGE": 30.0, "XLARGE": 60.0}[self.network_class]
                
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
                        c, sol, _, sims_done, worker_basins = res.get()
                        if sol is not None: epoch_results.append((c, sol))
                        global_failed_basins.update(worker_basins)
                    except Exception as e:
                        print(f"     [Error] Worker {wid+1} crashed: {e}")

                if shared_progress is not None:
                    for wid in range(self.n_workers):
                        prog = shared_progress.get(wid, {})
                        if isinstance(prog, dict):
                            cumulative_epoch_sims += prog.get('sims', 0)

            else:
                for t in tasks:
                    try:
                        c, sol, _, sims_done, worker_basins = self.worker_task(t)
                        if sol is not None: epoch_results.append((c, sol))
                        global_failed_basins.update(worker_basins)
                        cumulative_epoch_sims += sims_done 
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
                archive_signatures = {self.pool.get_basin_signature(epoch_results_sorted[0][1])}

                for cost, sol in epoch_results_sorted[1:]:
                    sig = self.pool.get_basin_signature(sol)
                    if sig not in archive_signatures:
                        diverse_archive.append((cost, sol))
                        archive_signatures.add(sig)
                    if len(diverse_archive) >= 4: break 
                
                for cost, sol in epoch_results_sorted[1:]:
                    if len(elite_archive) + len(diverse_archive) >= 6: break
                    if (cost, sol) not in diverse_archive:
                        elite_archive.append((cost, sol))

                global_archive = elite_archive + diverse_archive

            if epoch < epochs - 1 and len(global_archive) >= 2:
                sigs = set()
                for _, sol in global_archive:
                    sigs.add(self.pool.get_basin_signature(sol))
                if len(sigs) == 1:
                    print(f"   [DIVERSITY ALARM] Archive collapsed to single basin! Injecting cold seeds.")
                    cold_seeds = self.seeder.make_diverse_seeds()
                    for cs in cold_seeds[:2]:
                        c, _, feas, _ = self.ctx.get_cached_stats(cs)
                        if feas: global_archive.append((c, cs))
                    global_archive = sorted(global_archive, key=lambda x: x[0])[:6]

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
        total_cluster_sims = self.ctx.sim_count + cumulative_epoch_sims

        if global_best_cost != float('inf'):
            print(f"\n[AnalyticalSolver] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Total Time: {total_time/60:.1f}m | Total Sims: {total_cluster_sims:,})")
        else:
            print(f"\n[AnalyticalSolver] EXECUTION ABORTED. No valid solutions.")
        
        real_diams = [self.ctx.diameters[i] for i in global_best_sol] if global_best_sol else []
        return real_diams