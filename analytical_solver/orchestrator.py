import math
import random
import time
import bisect
import itertools
import multiprocessing
import networkx as nx
import numpy as np

try:
    from .fast_math import fast_avg_hamming
    HAS_FAST_MATH = True
except ImportError:
    HAS_FAST_MATH = False

from .context import SolverContext
from .pool import SolutionPool
from .local_search import LocalSearch
from .worker import WorkerTask, run_island_worker
from .acceptance import AcceptancePolicy, Verdict
from .adaptive_rates import AdaptiveRates
from .strategy_selector import StrategySelector
from .kick_memory import KickMemory
from .kick_registry import (
    FALLBACK_KICK,
    KICK_REGISTRY,
    TRACKED_STRATEGIES,
    pool_for,
)

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
                squeezed = self.ls.gradient_squeeze(healed, locked_pipes=frozen, max_passes=2, quick_mode=True)
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
                
            _, p, _, _ = self.ctx.get_cached_stats(idx_sol)
            if p < self.ctx.h_min:
                idx_sol, is_healed, _ = self.ls.heal_network(idx_sol, set())
                if not is_healed: continue 
            squeezed_sol = self.ls.gradient_squeeze(idx_sol, max_passes=12, quick_mode=True)
            seeds.append(squeezed_sol)
        return seeds
    
    def make_diverse_seeds_for_fast(self):
        seeds = []
        velocities = [x / 10.0 for x in range(5, 21)]
        
        for v in velocities:
            idx_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            for _ in range(5):
                real_diams = [self.ctx.diameters[i] for i in idx_sol]
                flows, _ = self.ctx.simulator.get_hydraulic_state(real_diams)
                new_idx = [self.calculate_ideal_d(q, v) for q in flows] 
                if new_idx == idx_sol: break
                idx_sol = new_idx
                
            _, p, _, _ = self.ctx.get_cached_stats(idx_sol)
            if p < self.ctx.h_min:
                idx_sol, is_healed, _ = self.ls.heal_network(idx_sol, set())
                if not is_healed: continue 
                
            squeezed_sol = self.ls.gradient_squeeze(idx_sol, max_passes=2, quick_mode=True)
            seeds.append(squeezed_sol)
            
        return seeds

    def make_backbone_seed(self, target_v=1.2):
        max_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
        flows, _ = self.ctx.simulator.get_hydraulic_state([self.ctx.diameters[i] for i in max_sol])

        flow_data = [(i, abs(q)) for i, q in enumerate(flows)]
        flow_data.sort(key=lambda x: x[1], reverse=True)

        n = self.ctx.num_pipes
        n_trunk = max(1, int(n * 0.20))
        n_periphery = max(1, int(n * 0.50))

        new_sol = list(max_sol)
        for rank, (p_idx, q) in enumerate(flow_data):
            if rank < n_trunk:
                new_sol[p_idx] = self.ctx.max_d_idx
            elif rank >= n - n_periphery:
                new_sol[p_idx] = 0
            else:
                new_sol[p_idx] = self.calculate_ideal_d(q, target_v)

        healed, ok, _ = self.ls.heal_network(new_sol, set())
        if ok:
            return self.ls.gradient_squeeze(healed, max_passes=2, quick_mode=True)
        return None

    
    def greedy_path_relink(self, sol_A, sol_B):
        current = list(sol_A)
        diff = [i for i in range(self.ctx.num_pipes) if sol_A[i] != sol_B[i]]
        
        random.shuffle(diff)
        
        best_intermediate = list(sol_A)
        best_cost, _, _, _ = self.ctx.get_cached_stats(sol_A)
        
        for pipe_idx in diff:
            test = list(current)
            test[pipe_idx] = sol_B[pipe_idx]
            
            c, p, _, _ = self.ctx.get_cached_stats(test)
            
            if p >= self.ctx.h_min:
                current = test
                
                if c < best_cost:
                    best_cost = c
                    best_intermediate = list(test)
                    
        return best_intermediate

class IslandWorker:
    def __init__(self, ctx, kicker, local_search, worker_id, n_workers, max_sims, beam_width, network_class, global_archive, ablation=False):
        self.ctx = ctx
        self.kicker = kicker
        self.ls = local_search
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.max_sims = max_sims
        self.BEAM_WIDTH = beam_width
        self.network_class = network_class
        self.global_archive = global_archive
        
        self.pool = SolutionPool(self.ctx)
        self.seeder = SeedFactory(self.ctx, self.ls, self.n_workers, self.BEAM_WIDTH)
        
        # Деревоподібність мережі не змінюється, тому обчислюється один раз,
        # а не на кожному застосуванні кіка.
        self._is_tree = nx.is_tree(self.ctx.base_G_flow)

        # Раніше цей перелік існував у трьох місцях (self.strategies,
        # all_tracked і три пули в _apply_kick) і містив стратегію
        # "SUBMARINE", якої немає в диспетчері — тобто їй марно велася
        # статистика UCB1. Тепер джерело одне — реєстр.
        # Ablation-режим: фіксує T, знімає послаблення вимоги тиску й
        # вимикає самонавчання. Раніше це була локальна константа в тілі
        # `_apply_kick`, тобто шість недосяжних гілок. Тепер це параметр,
        # і ablation-прогін не вимагає правки вихідника.
        self.ablation = ablation
        self.selector = StrategySelector(TRACKED_STRATEGIES, ablation=ablation)
        self.acceptance = AcceptancePolicy(ablation=ablation)
        
        self.SINGLE_CANDIDATES = self.ctx.profile.single_candidates
        
        self.stagnation_counter = 0
        self.run_best_cost = float('inf')
        self.run_best_sol = None
        self.global_best_cost = float('inf')
        self.last_published_cost = float('inf')
        self.progress_ratio = 0.0
        self.is_late_game = False
        
        self.last_injected_peer = {}
        self.rescue_fired = False
        self._force_flush_next = False
        self.ipc_immunity = 30
        # Чотири короткі пам'яті кіків, які створювались, очищались і
        # передавались у кік разом — тепер один об'єкт.
        self.kick_memory = KickMemory()
        
        # Розміри кроку кіків і правила їх навчання (JADE) — раніше
        # чотири поля, які оновлювались у п'яти місцях `_apply_kick`,
        # кожне через власну копію ланцюга if/elif по стратегії.
        self.rates = AdaptiveRates()

        # Ці поля читалися дев'ятьма методами (`base_dyn_bonus`), п'ятьма
        # (`stag_limit`) і одним (`reserve_pool`), але присвоювалися лише в
        # `run()` — тобто до її виклику об'єкт був у невалідному стані, і
        # будь-яке звернення до методу давало AttributeError у глибині кіка.
        # `run()` перераховує їх перед першим кіком і далі щороунду, тому ці
        # значення ніколи не доходять до пошуку: вони існують, щоб об'єкт був
        # валідним з моменту створення (і щоб методи можна було тестувати).
        self.base_dyn_bonus = 0.0
        self.stag_limit = 4
        self.reserve_pool = []
        self._last_ping_sims = 0
        self._last_ping_time = 0.0
        self._migration_interval_sims = 15000
        self._last_migration_sims = 0

    def progress_of(self, sims, elapsed, time_budget):
        """Частка пройденого прогону — за тим бюджетом, який ближче до кінця.

        Раніше це було «або симуляції, або час»: при заданому `max_sims`
        час не враховувався взагалі. Прогін, обмежений годинником (а на
        великих мережах практично будь-який: квота XLARGE — 30 млн
        симуляцій, тобто близько 20 діб на одному ядрі), давав частку
        близько нуля до самого кінця. А від неї залежать **одинадцять**
        рішень: перехід у другу половину прогону (`is_late_game`), ліміт
        застою, інтервал міграції, поріг «авантюриста», вікно
        серцебиття, заморозка консенсусу, дистанція різноманітності
        променя, його динамічна ширина. Тобто такий прогін від початку
        до кінця поводився так, ніби щойно почався.

        Тепер береться максимум з двох часток. Там, де скінченний лише
        один бюджет, результат не змінюється: другий доданок дає 0.
        """
        by_sims = (sims / max(1, self.max_sims)
                   if self.max_sims != float('inf') else 0.0)
        by_time = (elapsed / time_budget
                   if time_budget and time_budget != float('inf') else 0.0)
        return min(1.0, max(by_sims, by_time))

    def run(self, time_budget, global_best_cost, shared_progress):
        self.global_best_cost = global_best_cost
        start_time = time.time()
        self.pool.clear_all()
        
        # Раніше тут була гілка під `epoch != 0`, яка звала `make_warm_seeds`
        # (96 рядків, CC 37). Епох більше немає — гілка була недосяжна.
        seeds = self.seeder.make_diverse_seeds()
        self.reserve_pool = self.seeder.make_reserve_pool(size=4)
        
        best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds]) if seeds else global_best_cost
        self.base_dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001 
        
        self._initialize_seeds(seeds)
        
        self.ipc_immunity = 30
        self.stag_limit = 4 
        round_idx = 0
        run_start_sims = self.ctx.sim_count
        just_flushed = False
        
        self._last_ping_sims = self.ctx.sim_count
        self._last_ping_time = time.time()
        
        if self.max_sims != float('inf'):
            if self.progress_ratio > 0.8:
                self._migration_interval_sims = max(1000, self.max_sims // 100) 
            else:
                self._migration_interval_sims = max(1000, int((self.max_sims // 20) * (1.0 - 0.6 * self.progress_ratio)))
        else:
            self._migration_interval_sims = 15000
            
        self._last_migration_sims = self.ctx.sim_count
        
        def update_progress_ping():
            if shared_progress is not None:
                curr_sims = self.ctx.sim_count
                curr_time = time.time()
                
                if (curr_sims - self._last_ping_sims) > 10000 or (curr_time - self._last_ping_time) > 2.0:
                    self._last_ping_sims = curr_sims
                    self._last_ping_time = curr_time
                    
                    try:
                        current_data = shared_progress.get(self.worker_id)
                        if isinstance(current_data, dict):
                            new_data = dict(current_data)
                            new_data['sims'] = curr_sims
                            shared_progress[self.worker_id] = new_data
                    except Exception:
                        pass
                        
        self.ls.progress_callback = update_progress_ping
        
        while True:
            self.pool.current_round = round_idx 
            self.base_dyn_bonus = min(self.run_best_cost, self.global_best_cost) * 0.001 * random.uniform(0.95, 1.05)
            
            gb = shared_progress.get('global_best') if shared_progress else None
            if shared_progress is not None:
                self._process_ipc(shared_progress, gb)
                self._check_rescue(gb)
                shared_progress[self.worker_id] = {"round": round_idx + 1, "sims": self.ctx.sim_count, "best_cost": self.run_best_cost}
            
            elapsed = time.time() - start_time
            sims_used = self.ctx.sim_count - run_start_sims
            if elapsed > time_budget or sims_used >= self.max_sims: break
                
            just_flushed = self.rescue_fired
            self.rescue_fired = False
            
            self.progress_ratio = self.progress_of(sims_used, elapsed, time_budget)
            self.is_late_game = self.progress_ratio > 0.5 
            
            self.stag_limit = max(4, int(12 * (1.0 - self.progress_ratio)))
            
            if self.max_sims != float('inf'):
                base_interval = self.max_sims // 20
                if self.progress_ratio > 0.8:
                    self._migration_interval_sims = max(1000, base_interval // 5)
                else:
                    self._migration_interval_sims = max(1000, int(base_interval * (1.0 - 0.6 * self.progress_ratio)))
            
            self._check_mini_restart(gb)
            if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()

            self._apply_swap(round_idx, shared_progress)

            if self.stagnation_counter >= 2:
                self._apply_kick(round_idx, shared_progress)
                
            if self._force_flush_next:
                just_flushed = True
                self._force_flush_next = False

            next_gen = self._generate_mutations()
            
            if next_gen:
                next_gen.sort(key=lambda x: x[0])
                just_flushed = self._beam_search_and_update(next_gen, just_flushed, shared_progress, round_idx)
            else:
                self._emergency_respawn()
                just_flushed = True

            curr_sims = self.ctx.sim_count
            if shared_progress is not None and (curr_sims - self._last_migration_sims) >= self._migration_interval_sims:
                self._last_migration_sims = curr_sims
                
                if 'global_archive' in shared_progress:
                    self.global_archive = shared_progress['global_archive']
                
                if gb is not None:
                    gb_cost, gb_sol = gb
                    
                    if self.run_best_cost > gb_cost * 1.005:
                        T = min(1.0, self.stagnation_counter / max(1, self.stag_limit * 3.0))
                        
                        if T < 0.3:
                            hybrid_sol = self._spatial_crossover(self.run_best_sol, gb_sol, T=0.25)
                            
                            peers = [i for i in range(self.n_workers) if i != self.worker_id]
                            random.shuffle(peers)
                            peer_sol = None
                            for peer_id in peers:
                                peer_data = shared_progress.get(f'best_sol_{peer_id}')
                                if peer_data and abs(peer_data[0] - gb_cost) > 0.1:
                                    peer_sol = peer_data[1]
                                    break
                            
                            noise_msg = ""
                            if peer_sol:
                                hybrid_sol = self._spatial_crossover(hybrid_sol, peer_sol, T=0.05)
                                noise_msg = " + Peer Noise"
                            
                            c, p, _, _ = self.ctx.get_cached_stats(hybrid_sol)
                            if p < self.ctx.h_min:
                                hybrid_sol, ok, _ = self.ls.heal_network(hybrid_sol, set())
                                if not ok: hybrid_sol = list(gb_sol) 
                                    
                            h_c, _, _, _ = self.ctx.get_cached_stats(hybrid_sol)
                            
                            self.pool.active_pool.insert(0, (h_c, h_c, hybrid_sol))
                            self.stagnation_counter = 0
                            self.ipc_immunity = 50
                            
                            self.ctx.log(f"    📡 [MIGRATION] Soft injection of Global Best hybrid{noise_msg} ({h_c/1e6:.4f}M$)")
                        
                        elif T < 0.7:
                            hybrid_sol = self._spatial_crossover(self.run_best_sol, gb_sol, T)
                            
                            c, p, _, _ = self.ctx.get_cached_stats(hybrid_sol)
                            if p < self.ctx.h_min:
                                hybrid_sol, ok, _ = self.ls.heal_network(hybrid_sol, set())
                                if ok: c, p, _, _ = self.ctx.get_cached_stats(hybrid_sol)
                                
                            # УВАГА: перевірки тиску тут немає й не було — гібрид
                            # приймається лише за вартістю. Поведінка збережена
                            # свідомо; це окреме рішення, не частина етапу 2.
                            if c < self.run_best_cost:
                                self.ctx.log(f" 🧬 [HYBRID SUCCESS] Created new hybrid solution: {c/1e6:.4f}M$")
                                self.run_best_cost = c
                                self.run_best_sol = hybrid_sol
                                self.pool.active_pool.insert(0, (c, c, hybrid_sol))
                                self.stagnation_counter = 0
                            else:
                                self.ctx.log(" 🧬 [HYBRID FAILED] Hybrid was too expensive. Drift continues.")
                                
                        else:
                            self.ctx.log(f" 🛡️ [EXPLORATION SHIELD] Ignored global best. Deep Random Walk (T={T:.2f}).")
            round_idx += 1
                
        if shared_progress is not None:
            try:
                current_data = shared_progress.get(self.worker_id, {})
                if isinstance(current_data, dict):
                    current_data['sims'] = self.ctx.sim_count
                    current_data['best_cost'] = self.run_best_cost
                    shared_progress[self.worker_id] = current_data
            except Exception:
                pass

        self.ls.progress_callback = None
        
        _, final_p, _, _ = self.ctx.get_cached_stats(self.run_best_sol)
        if final_p < self.ctx.h_min:
            self.ctx.log(" ⚠️ [FINAL GUARD] Final solution strictly invalid. Applying hard heal.")
            healed_sol, ok, _ = self.ls.heal_network(self.run_best_sol, set())
            if ok:
                self.run_best_sol = healed_sol
                self.run_best_cost, _, _, _ = self.ctx.get_cached_stats(healed_sol)
            else:
                self.run_best_cost = float('inf')
                
        return self.run_best_cost, self.run_best_sol

    def _initialize_seeds(self, seeds):
        valid_sols = []
        for s in seeds:
            c, p, _, _ = self.ctx.get_cached_stats(s)
            score = self.ctx.score(c, p, self.base_dyn_bonus)
            
            self.pool.active_pool.append((score, c, s))
            self.pool.add_to_tabu(s, c)
            
            if p >= self.ctx.h_min:
                valid_sols.append((c, s))
                
        if not self.pool.active_pool:
            self.ctx.log("   > [WARNING] No valid seeds received. Injecting emergency max-diameter seed.")
            emergency = [self.ctx.max_d_idx] * self.ctx.num_pipes
            c, p, _, _ = self.ctx.get_cached_stats(emergency)
            
            self.pool.active_pool.append((c, c, emergency))
            if p >= self.ctx.h_min:
                valid_sols.append((c, emergency))
                
        if valid_sols:
            self.run_best_cost = min(x[0] for x in valid_sols)
            self.run_best_sol = next(x[1] for x in valid_sols if x[0] == self.run_best_cost)
        else:
            best_seed = min(self.pool.active_pool, key=lambda x: x[1])
            self.run_best_cost = best_seed[1] 
            self.run_best_sol = best_seed[2]
            
        self.ctx.log(f"   > Seeds initialized. Baseline Target: {self.run_best_cost/1e6:.4f}M$")

    def _process_ipc(self, shared_progress, gb):
        if self.ipc_immunity > 0:
            self.ipc_immunity -= 1
            return
        
        if gb and gb[0] < self.global_best_cost:
            self.global_best_cost = gb[0]
            self.ctx.log(f"   > [IPC] 📡 Received new Global Bound from peer: {self.global_best_cost/1e6:.4f}M$")
        
        for i in range(self.n_workers):
            if i == self.worker_id: continue
            peer = shared_progress.get(f'best_sol_{i}')
            last_cost = self.last_injected_peer.get(i, float('inf'))
            
            if peer and peer[0] < self.run_best_cost * 0.995 and peer[0] < last_cost - 1.0:
                is_adventurer = (self.n_workers >= 4 and self.worker_id == self.n_workers - 1)
                if is_adventurer and self.progress_ratio < 0.85:
                    continue 
                    
                is_massively_better = peer[0] < self.run_best_cost * 0.98
                if self.stagnation_counter < self.stag_limit and not is_massively_better:
                    continue 
                    
                peer_sol = list(peer[1])
                c_p, p_p, _, _ = self.ctx.get_cached_stats(peer_sol)
                if p_p >= self.ctx.h_min and not self.pool.is_basin_tabu(peer_sol):
                    score = self.ctx.score(c_p, p_p, self.base_dyn_bonus)
                    self.pool.active_pool.append((score, c_p, peer_sol))
                    self.ctx.log(f"     [IPC] 💉 Passively injected peer W{i+1} solution ({c_p/1e6:.4f}M$)")
                    self.last_injected_peer[i] = peer[0]

    def _check_rescue(self, gb):
        
        if self.ipc_immunity > 0:
            return
        
        global_lag = (self.run_best_cost - self.global_best_cost) / max(self.global_best_cost, 1.0)
        lag_threshold = 0.005 if self.is_late_game else 0.02 
        
        if (global_lag > lag_threshold and self.stagnation_counter >= self.stag_limit * 2) or global_lag > 0.05:
            if gb and gb[1]:
                self.ctx.log(f"   [RESCUE] Worker lagging by {global_lag:.1%}. Abandoning dead basin and adopting Global Best {gb[0]/1e6:.4f}M$!")
                
                gb_sol = list(gb[1])
                c_gb, p_gb, _, _ = self.ctx.get_cached_stats(gb_sol)
                if p_gb >= self.ctx.h_min:
                    self.run_best_cost = c_gb
                    self.run_best_sol = gb_sol
                    
                    score = self.ctx.score(c_gb, p_gb, self.base_dyn_bonus)
                    
                    self.pool.active_pool.insert(0, (score - 1e6, c_gb, gb_sol))
                    self.stagnation_counter = 0
                    self.pool.kick_tabu_set.clear()
                    self.rescue_fired = True

    def _check_mini_restart(self, gb):
        base_mult = self.ctx.profile.restart_patience
        effective_mult = base_mult if not self.is_late_game else max(4, base_mult // 2)
        
        if self.stagnation_counter >= self.stag_limit * effective_mult:
            self.ctx.log(f"   [MINI-RESTART] {self.stagnation_counter} rounds without progress. Soft restart.")
            self.pool.tabu_fingerprints.clear()
            self.pool.kick_tabu_set.clear()
            self.pool.active_pool.clear()
            
            if gb and gb[1]:
                archive_sol = list(gb[1])
                new_seeds = []
                
                base_perturb = max(10, self.ctx.num_pipes // 5)
                n_perturb = min(
                    self.ctx.num_pipes,
                    min(30, base_perturb) if self.is_late_game else min(45, base_perturb + 10)
                )
                
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
                    actually_added = 0
                    for seed in new_seeds:
                        c, p, _, _ = self.ctx.get_cached_stats(seed)
                        if p >= self.ctx.h_min:
                            score = self.ctx.score(c, p, self.base_dyn_bonus)
                            self.pool.active_pool.append((score, c, seed))
                            actually_added += 1
                            
                    if actually_added > 0:
                        self.stagnation_counter = 0
                        self.ctx.log(f"   [MINI-RESTART] Injected {actually_added} fresh seeds. Tabu cleared.")
                    else:
                        self.ctx.log("   [MINI-RESTART] All seeds infeasible. Stagnation persists.")

    def _apply_swap(self, round_idx, shared_progress):
        if round_idx > 0 and round_idx % 8 == 0:
            swapped = self.ls.swap_search(self.run_best_sol, self.base_dyn_bonus)
            c, p, _, _ = self.ctx.get_cached_stats(swapped)
            if p >= self.ctx.h_min and c < self.run_best_cost:
                is_ghost = False
                if (self.run_best_cost - c) > (self.run_best_cost * 0.02):
                    is_ghost = self.ctx.is_ghost_solution(swapped)
                    
                if is_ghost:
                    self.ctx.log(f"   > [SHIELD] Swap illusion blocked ({c/1e6:.4f}M$)!")
                else:
                    diff = self.run_best_cost - c
                    self.run_best_cost, self.run_best_sol = c, swapped
                    self.ctx.log(f"   > [SWAP] 💎 Micro-Optimization: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                    self._update_global_best(shared_progress)
                    
                    self.stagnation_counter = max(0, self.stagnation_counter - 1)
                    self.pool.kick_tabu_set.clear()
                    
                    swap_score = self.ctx.score(c, p, self.base_dyn_bonus) - (self.run_best_cost * 0.1)
                    self.pool.active_pool.insert(0, (swap_score, c, swapped))
    
    #: Частка рекорду, більше за яку виграш вважається суттєвим і повністю
    #: скидає застій. Менший виграш лише зменшує лічильник на 2.
    RECORD_RESET_FRACTION = 0.005

    @staticmethod
    def _stagnation_after_record(diff, new_best_cost, current_counter):
        """Новий лічильник застою після оновлення рекорду.

        Порог рахується від **нового** рекорду, а не від старого — так було
        в обох місцях, де це правило застосовується, і так лишається.
        Різниця між двома варіантами — 0.5 %, тому прогоном вона невидима;
        зафіксована тестом `test_stagnation_threshold_is_measured_against_
        the_new_best_not_the_old`.
        """
        if diff > new_best_cost * IslandWorker.RECORD_RESET_FRACTION:
            return 0
        return max(0, current_counter - 2)

    def _apply_kick(self, round_idx, shared_progress):
        real_T = min(1.0, self.stagnation_counter / max(1, self.stag_limit * 3.0))
        T = self.acceptance.effective_temperature(real_T)
        
        is_late_game = self.progress_ratio >= 0.5
        pool_strats = pool_for(real_T, len(self.global_archive), self._is_tree)

        choice = self.selector.pick(pool_strats, real_T, T, is_late_game)
        strategy, use_bandit = choice.strategy, choice.bandit_active

        self.ctx.log(f"{choice.mode_label} (Stag: {self.stagnation_counter}/"
                     f"{self.stag_limit}) -> Applying '{strategy}'...")
 
        if real_T >= 0.8 and self.pool.active_pool:
            source_pool = [x for x in self.pool.active_pool if self.ctx.get_cached_stats(x[2])[1] >= self.ctx.h_min]
            if not source_pool: source_pool = self.pool.active_pool
            kick_target = max(source_pool, key=lambda x: self.pool.hamming_distance(x[2], self.run_best_sol))[2]
        elif real_T >= 0.4 and self.pool.active_pool:
            kick_target = random.choice([x[2] for x in self.pool.active_pool])
        else:
            kick_mode = round_idx % 3
            if kick_mode == 0: kick_target = self.run_best_sol
            elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
            else: kick_target = random.choice([x[2] for x in self.pool.active_pool]) if self.pool.active_pool else self.run_best_sol
 
        spec = KICK_REGISTRY.get(strategy) or KICK_REGISTRY[FALLBACK_KICK]

        kick_args = {
            'T': T,
            **self.kick_memory.as_kick_args(strategy),
            'current_round': round_idx,
            'tabu_set': self.pool.kick_tabu_set,
            'dyn_bonus': self.base_dyn_bonus,
            'global_archive': self.global_archive,
            'max_perturb': int(self.ctx.num_pipes * 0.50),
            **self.rates.as_kick_args(),
        }

        try:
            res = getattr(self.kicker, spec.method)(kick_target, **kick_args)
        except Exception as e:
            self.ctx.log(f"      -> Error in {strategy}: {e}")
            return

        forced_sol, locked, log_msg = res.solution, res.locked, res.message
        path_sig, used_pct = res.path_signature, res.used_pct

        if res.failed:
            self.selector.note_failure(strategy)
            if res.failed_pipe is not None:
                self.kick_memory.note_failure(strategy, res.failed_pipe, round_idx)

            reason = log_msg if log_msg else "No targets"
            self.ctx.log(f"      -> Kick '{strategy}' failed. Reason: {reason}")
            
            if strategy == "TOPO_INV" and "All paths tabu" in reason:
                self.pool.kick_tabu_set.clear()
                self.ctx.log("      -> [TABU CLEARED] Topo paths memory reset to unblock strategy.")
            
            if use_bandit:
                self.selector.penalise(strategy, StrategySelector.FAILED_KICK_PENALTY)
            return
 
        self.selector.note_success(strategy)
 
        c, p, _, _ = self.ctx.get_cached_stats(forced_sol)
        base_h_min = self.ctx.h_min
        
        explosion_threshold = self.acceptance.explosion_threshold(self.run_best_cost, T)

        if self.acceptance.is_hopeless_pressure(base_h_min, p, T):
            self.selector.note_failure(strategy)
            return
 
        if p < self.acceptance.relaxed_h_min(base_h_min, T):
            heal_locks = set() if strategy in ["SPATIAL_PERTURB", "SMART_PERTURB", "RUIN_RECREATE"] else locked
            forced_sol, ok, boosts = self.ls.heal_network(forced_sol, heal_locks)
            if not ok: 
                if boosts > 0: self.ctx.log("     -> Heal Aborted Early: Cost exceeded limit.")
                return
 
        if log_msg: self.ctx.log(f"     -> {log_msg}")
 
        if strategy == "BASIN_ESCAPE":
            final_sol = forced_sol
            # Записується ПІДПИС басейну, а не вектор рішення: is_basin_tabu()
            # шукає саме підпис, і для мереж > 50 труб це різні речі.
            if self.run_best_sol: self.pool.add_basin_to_tabu(self.run_best_sol)
        else:
            quick_passes = max(1, 2 - int(T * 2))
            locked_for_squeeze = set() if strategy in ["SMART_PERTURB", "SPATIAL_PERTURB", "RUIN_RECREATE"] else locked
            quick_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked_for_squeeze, max_passes=quick_passes, quick_mode=True, dyn_bonus=self.base_dyn_bonus)
            quick_st = self.ctx.get_cached_stats(quick_sol)
            quick_cost = quick_st.cost
 
            hb_margin = self.acceptance.heartbeat_margin(self.progress_ratio)
            if (quick_st.is_feasible(base_h_min)
                    and quick_cost < self.run_best_cost * (1.0 + hb_margin)):
                gap = (quick_cost - self.run_best_cost) / max(self.run_best_cost, 1.0)
                deep_passes = self.acceptance.deep_passes(gap, T, self.ctx.profile)
 
                consensus_locked = set()
                if len(self.global_archive) >= 3 and is_late_game and T < 0.2 and not self._force_flush_next:
                    arch_sols = [x[1] for x in self.global_archive]
                    raw_consensus = {i for i in range(self.ctx.num_pipes) if all(sol[i] == arch_sols[0][i] for sol in arch_sols)}
                    max_frozen = 0 if self.progress_ratio > 0.88 else max(5, int(self.ctx.num_pipes * (0.25 if self.progress_ratio > 0.6 else 0.10)))
                    if len(raw_consensus) > max_frozen > 0:
                        interesting = [i for i in raw_consensus if 0 < arch_sols[0][i] < self.ctx.max_d_idx]
                        consensus_locked = set(list(interesting)[:max_frozen])
                    elif max_frozen > 0: consensus_locked = raw_consensus
 
                safe_consensus = consensus_locked - (locked if locked else set())
                final_locked = locked_for_squeeze.union(safe_consensus)
                final_sol = self.ls.gradient_squeeze(quick_sol, locked_pipes=final_locked, max_passes=deep_passes, quick_mode=(gap >= 0), dyn_bonus=self.base_dyn_bonus)
            else:
                final_sol = quick_sol
 
        st = self.ctx.get_cached_stats(final_sol)
        c, p = st.cost, st.p_min
        deficit = max(0.0, base_h_min - p) if st.simulation_ok else float('inf')

        if st.simulation_ok and deficit <= self.acceptance.deficit_tolerance(T):
            effective_cost = self.acceptance.effective_cost(c, deficit, self.run_best_cost)
            
            if effective_cost > explosion_threshold: 
                self.ctx.log(f"      -> Hard-Rejected (Cost Explosion): {effective_cost/1e6:.4f}M$")
                
                if use_bandit:
                    self.selector.penalise(strategy, StrategySelector.EXPLOSION_PENALTY)
                
                if used_pct is not None and not self.ablation:
                    self.rates.shrink_after_explosion(strategy)
                    self.ctx.log(f"         🧠 [JADE] Shrunk {strategy} size due to explosion.")
                return
        else: 
            return
 
        score = self.ctx.score(effective_cost, p, self.base_dyn_bonus) if deficit == 0 else effective_cost

        # Рішення приймає `AcceptancePolicy`; застосування лишається тут,
        # дослівно в тому порядку, що й було (§20.2). `is_basin_tabu` і
        # `is_ghost` передаються функціями, бо в оригіналі обчислювались
        # лише в тих гілках, куди доходив `elif`, а `is_ghost_solution`
        # ще й виконує симуляції.
        verdict = self.acceptance.classify(
            is_escape=(strategy == "BASIN_ESCAPE"),
            effective_cost=effective_cost, deficit=deficit,
            best_cost=self.run_best_cost, temperature=T,
            is_basin_tabu=lambda: self.pool.is_basin_tabu(final_sol),
            is_ghost=lambda: self.ctx.is_ghost_solution(final_sol),
        )

        if verdict == Verdict.GHOST_VETO:
            return

        if verdict == Verdict.RECORD:
            diff = self.run_best_cost - effective_cost
            self.run_best_cost, self.run_best_sol = effective_cost, final_sol
            self.pool.active_pool.insert(0, (score, effective_cost, final_sol))
 
            self.stagnation_counter = self._stagnation_after_record(
                diff, self.run_best_cost, self.stagnation_counter)
 
            if use_bandit:
                improvement_pct = diff / max(self.run_best_cost, 1.0)
                self.selector.reward(strategy, StrategySelector.record_reward(improvement_pct))
                self.selector.decay_others(strategy, StrategySelector.OTHERS_DECAY_ON_RECORD)
 
            if used_pct is not None and not self.ablation:
                self.rates.learn(strategy, used_pct, self.rates.record_alpha(used_pct))
                    
            self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
            self._update_global_best(shared_progress)

        elif verdict == Verdict.ESCAPE:
            self.pool.active_pool.clear()
            self.pool.active_pool.append((score, effective_cost, final_sol))
            self.stagnation_counter = 0
            self.kick_memory.clear()
            self._force_flush_next = True
            self.run_best_cost = effective_cost
            self.run_best_sol = list(final_sol)
            self.ipc_immunity = 50
            if effective_cost < self.global_best_cost: self._update_global_best(shared_progress)
            self.ctx.log(f"      -> POOL FLUSHED. Adopted Major Escape: {effective_cost/1e6:.4f}M$")
 
            self.selector.reset_after_escape()

            if used_pct is not None and not self.ablation:
                self.rates.learn_escape(used_pct)
 
        elif verdict == Verdict.POOL:
            pool_gap = verdict.gap
            self.pool.active_pool.append((score * 1.05, effective_cost, final_sol))
            if use_bandit:
                hamming_dist = self.pool.hamming_distance(final_sol, self.run_best_sol)
                diversity_ratio = hamming_dist / self.ctx.num_pipes
                if T >= 0.2 and diversity_ratio > 0.02:
                    self.selector.reward(
                        strategy, StrategySelector.diversity_reward(diversity_ratio, T))
            
            if used_pct is not None and not self.ablation:
                self.rates.learn(strategy, used_pct, self.rates.pool_alpha(pool_gap))

            if res.failed_pipe is not None:
                self.kick_memory.note_boost(strategy, res.failed_pipe, round_idx)

            self.ctx.log(f"      -> Added to Pool (Water Level Accept): {effective_cost/1e6:.4f}M$")
 
        elif verdict == Verdict.HAIL_MARY:
            quick = self.ls.gradient_squeeze(final_sol, max_passes=1, quick_mode=True, dyn_bonus=self.base_dyn_bonus)
            quick_hail = self.ctx.get_cached_stats(quick)
            q_c = quick_hail.cost
 
            if quick_hail.is_feasible(base_h_min) and q_c < self.run_best_cost:
                diff = self.run_best_cost - q_c
                self.run_best_cost, self.run_best_sol = q_c, quick
                self.pool.active_pool.insert(0, (q_c, q_c, quick))
                self.stagnation_counter = self._stagnation_after_record(
                    diff, self.run_best_cost, self.stagnation_counter)
                
                if use_bandit:
                    self.selector.reward(strategy, StrategySelector.record_reward(
                        diff / max(self.run_best_cost, 1.0)))
                
                self._update_global_best(shared_progress)
                self.ctx.log(f"     -> 🚀 HAIL MARY → DIRECT RECORD: {q_c/1e6:.4f}M$")
            else:
                self.pool.active_pool.append((score * 1.20, effective_cost, final_sol))
                self.stagnation_counter = int(self.stag_limit * 1.8) 
                self.ctx.log(f"     -> 🚀 HAIL MARY ACCEPT: {effective_cost/1e6:.4f}M$")
        else:
            self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {effective_cost/1e6:.4f}M$")
            
            if used_pct is not None and not self.ablation:
                self.rates.shrink_after_reject(strategy)
 
        if path_sig: self.pool.kick_tabu_set.add(path_sig)
        self.kick_memory.note_zero_sum(strategy, locked, round_idx)

    def _generate_mutations(self):
        next_gen = []
        for _, _, parent_sol in self.pool.active_pool:
            unit_losses = self.ctx.get_cached_heuristics(parent_sol)
            high_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i], reverse=True)
            low_friction = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i]) 
            
            st = self.ctx.get_cached_stats(parent_sol)
            parent_p_surplus = st.surplus(self.ctx.h_min) if st.simulation_ok else 0.0
            
            if parent_p_surplus > 10.0: downgrade_limit, upgrade_limit = 15, self.SINGLE_CANDIDATES
            elif parent_p_surplus < 2.0: downgrade_limit, upgrade_limit = 3, self.SINGLE_CANDIDATES // 2
            else: downgrade_limit, upgrade_limit = 8, self.SINGLE_CANDIDATES
            
            if self.ctx.profile.is_large:
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
    
    def _spatial_crossover(self, base_sol, donor_sol, T):
        import random
        import networkx as nx
        
        if not hasattr(self.ctx, 'base_G_flow'):
            return [donor_sol[i] if random.random() < 0.5 else base_sol[i] for i in range(self.ctx.num_pipes)]

        G = self.ctx.base_G_flow
        radius = 2 + int(6.0 * T) 
        
        epicenter = random.choice(list(G.nodes()))
        local_nodes = set(nx.single_source_shortest_path_length(G, epicenter, cutoff=radius).keys())
        
        local_pipes = set()
        
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
                            
        if not local_pipes:
            fallback_size = max(1, int(self.ctx.num_pipes * 0.15))
            local_pipes = set(random.sample(range(self.ctx.num_pipes), fallback_size))
                
        new_sol = list(base_sol)
        transferred_count = 0
        
        for p_idx in local_pipes:
            if new_sol[p_idx] != donor_sol[p_idx]:
                new_sol[p_idx] = donor_sol[p_idx]
                transferred_count += 1
                
        self.ctx.log(f"      🧬 [CROSSOVER] Transferred {transferred_count} pipes from Global Best (R={radius}).")
        return new_sol

    def _beam_search_and_update(self, next_gen, just_flushed, shared_progress, round_idx):
        unique_next_pool = []
        found_new_record = False
        max_d = max(2, int(self.ctx.num_pipes * 0.05)) 
        min_dist = max(1, int(max_d * ((1.0 - self.progress_ratio) ** 3)))
        
        for rank, (score, cost, sol) in enumerate(next_gen):
            if self.pool.is_tabu(sol, cost): continue
            if len(unique_next_pool) < self.BEAM_WIDTH:
                if rank == 0: 
                    is_deep = (round_idx % 2 == 0) or just_flushed
                    refined_sol = self.ls.gradient_squeeze(sol, max_passes=5, quick_mode=not is_deep, dyn_bonus=self.base_dyn_bonus, min_rel_improvement=0.0005)
                elif rank < 3: 
                    refined_sol = self.ls.gradient_squeeze(sol, max_passes=3, quick_mode=True, dyn_bonus=self.base_dyn_bonus, min_rel_improvement=0.002)
                else: 
                    refined_sol = sol

                is_diverse = True
                for _, _, peer_sol in unique_next_pool:
                    if self.pool.hamming_distance(refined_sol, peer_sol) < min_dist:
                        is_diverse = False; break
                        
                if not is_diverse and not just_flushed: continue

                st = self.ctx.get_cached_stats(refined_sol)
                real_cost = st.cost
                if st.simulation_ok:
                    real_p_surplus = st.surplus(self.ctx.h_min)
                    is_strictly_valid = (real_p_surplus >= 0.0)
                    is_epsilon_valid = (real_p_surplus >= -0.5) 
                    
                    if is_strictly_valid or is_epsilon_valid:
                        if is_strictly_valid:
                            score = self.ctx.score(real_cost, st.p_min, self.base_dyn_bonus)
                        else:
                            score = real_cost + (abs(real_p_surplus) * self.base_dyn_bonus * 50.0)
                        
                        if is_strictly_valid and real_cost < self.run_best_cost:
                            is_ghost = False
                            if (self.run_best_cost - real_cost) > (self.run_best_cost * 0.02):
                                is_ghost = self.ctx.is_ghost_solution(refined_sol)
                                
                            if is_ghost:
                                self.ctx.log(f"   > [SHIELD] Beam illusion blocked ({real_cost/1e6:.4f}M$).")
                                continue 

                            diff = self.run_best_cost - real_cost
                            self.run_best_cost, self.run_best_sol = real_cost, refined_sol
                            
                            if diff > (self.run_best_cost * 0.005):
                                found_new_record = True 
                                self.ctx.log(f"   > [R{round_idx+1}] 💎 New RECORD: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                            else:
                                self.stagnation_counter = max(0, self.stagnation_counter - 2)
                                self.ctx.log(f"   > [R{round_idx+1}] 💎 Micro-Step: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                            
                            self._update_global_best(shared_progress)

                        unique_next_pool.append((score, real_cost, refined_sol))
                        self.pool.add_to_tabu(refined_sol, real_cost)

        if unique_next_pool:
            unique_next_pool.sort(key=lambda x: x[0])
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
        else:
            self.ctx.log("     [BEAM] No valid children generated. Keeping current pool.")
            self.stagnation_counter += 1
            return False

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
                st = self.ctx.get_cached_stats(fresh)
                if st.simulation_ok:
                    score = self.ctx.score(st.cost, st.p_min, self.base_dyn_bonus)
                    self.pool.active_pool[-1] = (score, st.cost, fresh)

    def _emergency_respawn(self):
        self.ctx.log("     [EMERGENCY] Beam search deadlocked (no valid/diverse children). Flushing pool & tabu!")
        self.pool.active_pool.clear()
        self.pool.tabu_fingerprints.clear()
        self.stagnation_counter += 1
        
        res1 = self.kicker.smart_perturbation_kick(self.run_best_sol, T=1.0)
        forced_rescue = res1[0] if (isinstance(res1, tuple) and len(res1) > 0) else None

        if forced_rescue is not None:
            st = self.ctx.get_cached_stats(forced_rescue)
            if st.is_feasible(self.ctx.h_min):
                self.pool.active_pool.append((st.cost - 1e6, st.cost, forced_rescue))
                self.stagnation_counter = 0 
        
        if not self.pool.active_pool:
            res2 = self.kicker.smart_perturbation_kick(self.run_best_sol, T=0.6)
            healed_seed = res2[0] if (isinstance(res2, tuple) and len(res2) > 0) else None

            if healed_seed is not None:
                st = self.ctx.get_cached_stats(healed_seed)
                if st.simulation_ok:
                    self.pool.active_pool.append((st.cost - 1e6, st.cost, healed_seed))
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
            if shared_progress is not None:
                current_gb = shared_progress.get('global_best', (float('inf'), []))
                if self.run_best_cost < current_gb[0]:
                    shared_progress['global_best'] = (self.run_best_cost, list(self.run_best_sol))


class AnalyticalSolver:
    def __init__(self, simulator_instance, available_diameters, v_opt=1.0, max_sims=None, time_limit_sec=None, pool=None, log_dir=None, n_workers=1):
        self.ctx = SolverContext(simulator_instance, available_diameters, v_opt)
        self.ls = LocalSearch(self.ctx)
        self.mp_pool = pool
        self.log_dir = log_dir
        self.n_workers = n_workers
        
        self.network_class = self.ctx.profile.name
        self.BEAM_WIDTH = self.ctx.profile.beam_width
        self.seeder = SeedFactory(self.ctx, self.ls, self.n_workers, self.BEAM_WIDTH)
        self.pool = SolutionPool(self.ctx) 
        
        self.BASE_SIM_BUDGET = self.ctx.profile.sim_budget
            
        if max_sims is not None:
            self.max_sims = max_sims
        else:
            self.max_sims = self.BASE_SIM_BUDGET
            
        if time_limit_sec is not None:
            self.time_limit_sec = time_limit_sec
        else:
            cluster_speed = max(self.ctx.sim_speed, 1.0) * self.n_workers
            self.time_limit_sec = (self.max_sims / cluster_speed) * 5
            
    def solve_fast(self, ui_callback=None):
        print("\n[AnalyticalSolver] ⚡ Initiating FAST Analytical Search (Velocity Sweep)...\n")
        start_time = time.time()
        
        print("   > Sweeping ideal velocities (0.5 to 2.0 m/s)...")
        seeds = self.seeder.make_diverse_seeds_for_fast ()
        
        print("   > Sweeping Backbone (Trunk-Branch) configurations...")
        for v in [0.8, 1.0, 1.2, 1.5, 1.8]:
            bb_seed = self.seeder.make_backbone_seed(target_v=v)
            if bb_seed:
                seeds.append(bb_seed)
                
        valid_seeds = []
        seen_sigs = set()
        
        for sol in seeds:
            c, p, _, _ = self.ctx.get_cached_stats(sol)
            if p >= self.ctx.h_min:
                sig = tuple(sol)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    valid_seeds.append((c, list(sol)))
                    
        global_best_cost = float('inf')
        global_best_sol = None
        
        if valid_seeds:
            valid_seeds.sort(key=lambda x: x[0])
            top_n = min(5, len(valid_seeds))
            
            print(f"   > Sweep finished. Found {len(valid_seeds)} valid seeds. Deep Polishing TOP-{top_n}...")
            
            for rank in range(top_n):
                raw_cost, raw_sol = valid_seeds[rank]
                
                polished = self.ls.gradient_squeeze(raw_sol, max_passes=None, quick_mode=False, dyn_bonus=raw_cost * 0.001)
                p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
                
                if p_p >= self.ctx.h_min and p_cost < global_best_cost:
                    global_best_cost = p_cost
                    global_best_sol = polished
                    print(f"     [{rank+1}/{top_n}] Polished {raw_cost/1e6:.4f}M$ ➡️ RECORD: {global_best_cost/1e6:.4f}M$")
                else:
                    print(f"     [{rank+1}/{top_n}] Polished {raw_cost/1e6:.4f}M$ ➡️ {p_cost/1e6:.4f}M$ (Discarded)")
                    
                if ui_callback is not None:
                    ui_callback(self.ctx.sim_count, global_best_cost, global_best_sol)
                    
        if global_best_sol is None:
            print("\n[WARNING] Fast seeds infeasible. Falling back to max diameters.")
            global_best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            global_best_cost, _, _, _ = self.ctx.get_cached_stats(global_best_sol)
            
        total_time = time.time() - start_time
        print(f"\n[Fast Analytical] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Time: {total_time:.2f}s | Sims: {self.ctx.sim_count})")
        
        self.history = [(self.ctx.sim_count, global_best_cost)]
        if ui_callback is not None:
            ui_callback(self.ctx.sim_count, global_best_cost, global_best_sol)
        
        real_diams = [self.ctx.diameters[i] for i in global_best_sol]
        return real_diams

    def solve_standalone(self, ui_callback=None):
        print("\n[AnalyticalSolver] ⚡ Initiating Continuous Island Model Search...\n")
        start_time = time.time()
        global_best_cost = float('inf')
        global_best_sol = None
        global_archive = []
        
        self.history = []

        # Епох більше немає: острівна модель із асинхронною синхронізацією
        # замінила їх на один прогін із міграцією (§2.3). Цикл `for epoch in
        # range(1)` і поле `epoch` прибрані, бюджет ділиться між островами.
        run_time_budget = self.time_limit_sec

        if self.max_sims == float('inf'):
            worker_sim_quota = float('inf')
        else:
            worker_sim_quota = int(self.max_sims // self.n_workers)

        quota_str = "∞" if worker_sim_quota == float('inf') else f"{worker_sim_quota:,}"
        print(f"  [Quota] Allocated {quota_str} sims per worker.\n")

        manager = multiprocessing.Manager() if self.mp_pool else None
        shared_progress = manager.dict() if manager else None
        if shared_progress:
            for i in range(self.n_workers): shared_progress[i] = 0
            shared_progress['global_archive'] = []

        # Симуляції, виконані островами: лічильник головного процесу їх
        # не бачить, бо кожен воркер має власний контекст.
        worker_sims_total = 0

        try:
            mode_str = "PARALLEL" if self.mp_pool else "SEQUENTIAL"
            print("="*46)
            print(f" [CONTINUOUS RUN] {mode_str} Workers: {self.n_workers} | Time Limit: {run_time_budget/60:.1f} min")
            print("="*46)
            
            seed_modifier = random.randint(1, 10000)
            tasks = [
                WorkerTask(
                    diameters=self.ctx.diameters, v_opt=self.ctx.v_opt,
                    time_budget=run_time_budget, global_best_cost=global_best_cost,
                    seed=seed_modifier + i, worker_id=i, n_workers=self.n_workers,
                    max_sims=worker_sim_quota, shared_progress=shared_progress,
                    log_dir=self.log_dir, global_archive=global_archive,
                )
                for i in range(self.n_workers)
            ]

            island_results = []
            
            if self.mp_pool:
                async_results = [
                    (t.worker_id, self.mp_pool.apply_async(run_island_worker, (t,)))
                    for t in tasks
                ]

                last_print_time = 0
                print_interval = self.ctx.profile.print_interval
                
                last_sent_cost = float('inf')
                
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
                        
                        live_best_sol = global_best_sol 
                        
                        live_archive = []
                        gb = shared_progress.get('global_best')
                        if gb: 
                            live_archive.append(gb)
                            if gb[0] < live_best: 
                                live_best = gb[0]
                                live_best_sol = gb[1]

                        for wid in range(self.n_workers):
                            prog = shared_progress.get(wid, 0)
                            if isinstance(prog, dict):
                                sims = prog.get('sims', 0)
                                w_best = prog.get('best_cost', float('inf'))
                                
                                if w_best < live_best: 
                                    live_best = w_best
                                    if 'best_sol' in prog:
                                        live_best_sol = prog['best_sol']
                            else:
                                sims = prog
                                
                            total_sims += sims
                            if sims > 0: status_parts.append(f"W{wid+1}:{sims//1000}k")
                            else: status_parts.append(f"W{wid+1}:--")
                            
                            w_sol = shared_progress.get(f'best_sol_{wid}')
                            if w_sol: live_archive.append(w_sol)
                                
                        status_str = " ".join(status_parts)
                        best_str = f"{live_best/1e6:.4f}M$" if live_best != float('inf') else "---"
                        print(f"   > [Live {m:02d}:{s:02d}] Best: {best_str} | Sims: {total_sims/1000:.1f}k | {status_str}")
                        
                        current_total = self.ctx.sim_count + worker_sims_total + total_sims
                        
                        if live_best < global_best_cost:
                            global_best_cost = live_best
                            global_best_sol = live_best_sol
                        
                        if live_best != float('inf'):
                            self.history.append((current_total, live_best))
                            
                        if ui_callback is not None:
                            if global_best_cost < last_sent_cost and global_best_sol is not None:
                                last_sent_cost = global_best_cost
                                ui_callback(current_total, global_best_cost, list(global_best_sol))
                            else:
                                ui_callback(current_total, global_best_cost, None)
                            
                        if live_archive:
                            live_archive.sort(key=lambda x: x[0])
                            unique_archive = []
                            seen_sigs = set()
                            for cost, sol in live_archive:
                                sig = tuple(sol)
                                if sig not in seen_sigs:
                                    seen_sigs.add(sig)
                                    unique_archive.append((cost, sol))
                                    if len(unique_archive) >= 6: break
                            shared_progress['global_archive'] = unique_archive
                    
                    time.sleep(1.0)

                for wid, res in async_results:
                    try:
                        result = res.get()
                        if result.best_solution is not None:
                            island_results.append((result.best_cost, result.best_solution))
                    except Exception as e:
                        print(f"     [Error] Worker {wid+1} crashed: {e}")

                if shared_progress is not None:
                    for wid in range(self.n_workers):
                        prog = shared_progress.get(wid, {})
                        if isinstance(prog, dict):
                            worker_sims_total += prog.get('sims', 0)

            else:
                for t in tasks:
                    try:
                        result = run_island_worker(t)
                        if result.best_solution is not None:
                            island_results.append((result.best_cost, result.best_solution))
                        worker_sims_total += result.sim_count
                        print(f"   > Worker {t.worker_id+1} Finished. Best: {result.best_cost/1e6:.4f}M$")
                    except Exception as e:
                        print(f"     [Error] Sequential Worker {t.worker_id+1} crashed: {e}")

        except KeyboardInterrupt:
            print("\n\n[AnalyticalSolver] 🛑 Отримано сигнал переривання (Ctrl+C)!")
            print("[AnalyticalSolver] М'яка зупинка. Перехід до генерації звітів...")

        # Результати воркерів, повернуті напряму. У паралельному режимі ті самі
        # значення приходять і через shared_progress, але в послідовному
        # (pool=None) це єдине джерело — без цього рішення воркера втрачалось
        # і solve_standalone повертав запасний варіант «усі максимальні».
        for c, sol in island_results:
            if sol is not None and c < global_best_cost:
                global_best_cost = c
                global_best_sol = list(sol)

        if shared_progress is not None:
            gb = shared_progress.get('global_best')
            if gb and gb[0] < global_best_cost:
                global_best_cost = gb[0]
                global_best_sol = list(gb[1])
                
            for wid in range(self.n_workers):
                w_sol = shared_progress.get(f'best_sol_{wid}')
                if w_sol and w_sol[0] < global_best_cost:
                    global_best_cost = w_sol[0]
                    global_best_sol = list(w_sol[1])

        print("\n[FINAL POLISH] Polishing global best solution...")
        if global_best_sol:
            polished = self.ls.gradient_squeeze(global_best_sol, max_passes=None, quick_mode=False, dyn_bonus=global_best_cost * 0.001)
            p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
            if p_p >= self.ctx.h_min and p_cost < global_best_cost:
                global_best_cost = p_cost
                global_best_sol = polished
                print(f"   > [POLISH] Improved! Final: {global_best_cost/1e6:.4f}M$")
                
            current_sims = self.ctx.sim_count + worker_sims_total
            self.history.append((current_sims, global_best_cost))
        else:
            print("\n[WARNING] No valid solution found. Returning safe default.")
            global_best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            try: global_best_cost, _, _, _ = self.ctx.get_cached_stats(global_best_sol)
            except: global_best_cost = float('inf')
        
        total_time = time.time() - start_time
        total_cluster_sims = self.ctx.sim_count + worker_sims_total

        if global_best_cost != float('inf'):
            print(f"\n[AnalyticalSolver] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Total Time: {total_time/60:.1f}m | Total Sims: {total_cluster_sims:,})")
        else:
            print("\n[AnalyticalSolver] EXECUTION ABORTED. No valid solutions.")
        
        if global_best_sol:
            real_diams = [self.ctx.diameters[max(0, min(i, self.ctx.max_d_idx))] for i in global_best_sol]
        else:
            real_diams = []
        return real_diams