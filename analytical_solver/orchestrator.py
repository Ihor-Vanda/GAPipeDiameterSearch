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
from .kicks import KickStrategies

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

    # def make_diverse_seeds(self):
    #     seeds = []
    #     for v in [1.0, 1.2, 0.8]:
    #         idx_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
    #         for _ in range(10):
    #             real_diams = [self.ctx.diameters[i] for i in idx_sol]
    #             flows, _ = self.ctx.simulator.get_hydraulic_state(real_diams)
    #             new_idx = [self.calculate_ideal_d(q, v) for q in flows] 
    #             if new_idx == idx_sol: break
    #             idx_sol = new_idx
                
    #         _, p, feas, _ = self.ctx.get_cached_stats(idx_sol)
    #         if not feas or p < self.ctx.simulator.config.h_min:
    #             idx_sol, is_healed, _ = self.ls.heal_network(idx_sol, set())
    #             if not is_healed: continue 
    #         squeezed_sol = self.ls.gradient_squeeze(idx_sol, max_passes=12, quick_mode=True)
    #         seeds.append(squeezed_sol)
    #     return seeds
    
    def make_diverse_seeds(self):
        """Створює масив зерен для різних цільових швидкостей (від 0.5 до 2.0 м/с)."""
        seeds = []
        # Генеруємо швидкості [0.5, 0.6, 0.7 ... 2.0]
        velocities = [x / 10.0 for x in range(5, 21)]
        
        for v in velocities:
            idx_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            for _ in range(5): # Достатньо 5 ітерацій для стабілізації потоку
                real_diams = [self.ctx.diameters[i] for i in idx_sol]
                flows, _ = self.ctx.simulator.get_hydraulic_state(real_diams)
                new_idx = [self.calculate_ideal_d(q, v) for q in flows] 
                if new_idx == idx_sol: break
                idx_sol = new_idx
                
            _, p, feas, _ = self.ctx.get_cached_stats(idx_sol)
            if not feas or p < self.ctx.simulator.config.h_min:
                idx_sol, is_healed, _ = self.ls.heal_network(idx_sol, set())
                if not is_healed: continue 
                
            # Робимо лише швидкий поверхневий сквіз (щоб не витрачати час)
            squeezed_sol = self.ls.gradient_squeeze(idx_sol, max_passes=2, quick_mode=True)
            seeds.append(squeezed_sol)
            
        return seeds

    def make_backbone_seed(self, target_v=1.2):
        """Будує 'скелет' (товсті магістралі, вузька периферія) для заданої швидкості."""
        max_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
        flows, _ = self.ctx.simulator.get_hydraulic_state([self.ctx.diameters[i] for i in max_sol])

        flow_data = [(i, abs(q)) for i, q in enumerate(flows)]
        flow_data.sort(key=lambda x: x[1], reverse=True)

        n = self.ctx.num_pipes
        n_trunk = max(1, int(n * 0.20))      # Топ 20%
        n_periphery = max(1, int(n * 0.50))  # Нижні 50%

        new_sol = list(max_sol)
        for rank, (p_idx, q) in enumerate(flow_data):
            if rank < n_trunk:
                new_sol[p_idx] = self.ctx.max_d_idx
            elif rank >= n - n_periphery:
                new_sol[p_idx] = 0
            else:
                new_sol[p_idx] = self.calculate_ideal_d(q, target_v) # Використовуємо передану швидкість

        healed, ok, _ = self.ls.heal_network(new_sol, set())
        if ok:
            return self.ls.gradient_squeeze(healed, max_passes=2, quick_mode=True)
        return None

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
            self.ctx.log(f"[CASTE] Worker {worker_id+1:02d} -> 🧬 RELINKER (Greedy Path-Relinking)")
            seeds.append(list(base_sol))
            
            if len(archive) >= 2:
                best_dist = -1
                target_sol = None
                for _, arch_sol in archive:
                    dist = sum(1 for a, b in zip(base_sol, arch_sol) if a != b)
                    if dist > best_dist and dist > 0:
                        best_dist = dist
                        target_sol = arch_sol
                
                diff_indices = [i for i in range(self.ctx.num_pipes) if base_sol[i] != target_sol[i]]
                
                if not diff_indices:
                    for _ in range(self.BEAM_WIDTH - 1):
                        seeds.append(list(base_sol))
                else:
                    self.ctx.log(f"   > Relinking across {len(diff_indices)} pipes to distant target.")
                    for _ in range(self.BEAM_WIDTH - 1):
                        child = self.greedy_path_relink(base_sol, target_sol)
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
    
    def greedy_path_relink(self, sol_A, sol_B):
        current = list(sol_A)
        diff = [i for i in range(self.ctx.num_pipes) if sol_A[i] != sol_B[i]]
        
        random.shuffle(diff)
        
        best_intermediate = list(sol_A)
        best_cost, _, _, _ = self.ctx.get_cached_stats(sol_A)
        
        for pipe_idx in diff:
            test = list(current)
            test[pipe_idx] = sol_B[pipe_idx]
            
            c, p, feas, _ = self.ctx.get_cached_stats(test)
            
            if feas and p >= self.ctx.simulator.config.h_min:
                current = test
                
                if c < best_cost:
                    best_cost = c
                    best_intermediate = list(test)
                    
        return best_intermediate

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
        
        # self.strategies = [
        #     "TOPO-DIV", "LOOP_BALANCE", "SYNC_TRIM", "FINISHER", "BOTTLENECK", 
        #     "ZERO_SUM", "SHOCK", "DIAM_DIVERSITY"
        # ]
        
        # if nx.is_tree(self.ctx.base_G_flow) and "LOOP_BALANCE" in self.strategies:
        #     self.strategies.remove("LOOP_BALANCE")
            
        # all_tracked = self.strategies + [
        #     "SEGMENT_RESTART", "IPC_CROSSOVER", "CORRIDOR_SEARCH", 
        #     "BASIN_ESCAPE", "ILS_PERTURBATION", "VNS_KICK", 
        #     "RUIN_RECREATE", "SUBMARINE"
        # ]
       

        self.strategies = ["SHOCK", "BOTTLENECK", "TOPO_INV", "LOOP_BALANCE", "ZERO_SUM", "TRIM"]
        
        if nx.is_tree(self.ctx.base_G_flow) and "LOOP_BALANCE" in self.strategies:
            self.strategies.remove("LOOP_BALANCE")
            
        all_tracked = self.strategies + [
            "SMART_PERTURB", "RUIN_RECREATE", "BASIN_ESCAPE", "SUBMARINE", "SPATIAL_PERTURB" # <--- Додано сюди
        ]
        
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
        self.bottleneck_failed_pipes = {}

    def run(self, time_budget, global_best_cost, shared_progress):
        # ==========================================
        # БЛОК 1: ІНІЦІАЛІЗАЦІЯ ВОРКЕРА ТА ПУЛІВ
        # ==========================================
        self.global_best_cost = global_best_cost
        start_time = time.time()
        self.pool.clear_all()
        
        # Ініціалізація або "розігрів" (якщо це не перша епоха)
        seeds = self.seeder.make_diverse_seeds() if self.epoch == 0 else self.seeder.make_warm_seeds(self.global_archive, self.worker_id)
        self.reserve_pool = self.seeder.make_reserve_pool(size=4)
        
        # Визначення базової вартості для розрахунку бонусів
        best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds]) if seeds else global_best_cost
        self.base_dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001 
        
        self._initialize_seeds(seeds)
        
        # ==========================================
        # БЛОК 2: НАЛАШТУВАННЯ ЛІЧИЛЬНИКІВ ТА ТРИГЕРІВ
        # ==========================================
        self.ipc_immunity = 30
        self.stag_limit = 4 
        round_idx = 0
        epoch_start_sims = self.ctx.sim_count 
        self._last_global_improvement_sim = self.ctx.sim_count
        just_flushed = False
        
        self._last_ping_sims = self.ctx.sim_count
        self._last_ping_time = time.time()
        
        # 🔴 АДАПТИВНИЙ ТРИГЕР МІГРАЦІЇ: Синхронізація ~20 разів за весь час роботи
        if self.max_sims != float('inf'):
            self._migration_interval_sims = max(1000, self.max_sims // 20)
        else:
            self._migration_interval_sims = 15000 # Fallback, якщо ліміту по симуляціях немає
            
        self._last_migration_sims = self.ctx.sim_count
        
        # Callback для оновлення прогрес-бару в GUI
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
        
        # ==========================================
        # БЛОК 3: ГОЛОВНИЙ ЦИКЛ ОПТИМІЗАЦІЇ (EVOLUTION LOOP)
        # ==========================================
        while True:
            self.pool.current_round = round_idx 
            self.base_dyn_bonus = min(self.run_best_cost, self.global_best_cost) * 0.001 * random.uniform(0.70, 1.30)
            
            # --- 3.1 Читання статусу від Оркестратора ---
            gb = shared_progress.get('global_best') if shared_progress else None
            if shared_progress is not None:
                self._process_ipc(shared_progress, gb)
                self._check_rescue(shared_progress, gb)
                shared_progress[self.worker_id] = {"round": round_idx + 1, "sims": self.ctx.sim_count, "best_cost": self.run_best_cost}
            
            # --- 3.2 Перевірка умов виходу ---
            elapsed = time.time() - start_time
            epoch_sims = self.ctx.sim_count - epoch_start_sims
            if elapsed > time_budget or epoch_sims >= self.max_sims: break
                
            just_flushed = self.rescue_fired
            self.rescue_fired = False
            
            # --- 3.3 Динамічна адаптація параметрів ---
            self.progress_ratio = min(1.0, epoch_sims / max(1, self.max_sims)) 
            self.is_late_game = self.progress_ratio > 0.5 
            self.stag_limit = 4 + int(4 * self.progress_ratio) 
            
            self._check_mini_restart(shared_progress, gb)
            if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()
            
            # ==========================================
            # БЛОК 4: ФАЗА МУТАЦІЙ ТА КІКІВ
            # ==========================================
            # Фаза 1: Локальний обмін генів між рішеннями в пулі (Crossover/Swap)
            self._apply_swap(round_idx, shared_progress)

            # Фаза 2: Глибокі структурні мутації (Kicks)
            if self.stagnation_counter >= 1 or round_idx % 2 == 0:
                self._apply_kick(round_idx, shared_progress, gb)
                
            if getattr(self, '_force_flush_next', False):
                just_flushed = True
                self._force_flush_next = False

            # Фаза 3: Генерація нащадків та звуження (Beam Search)
            next_gen = self._generate_mutations()
            
            if next_gen:
                next_gen.sort(key=lambda x: x[0])
                just_flushed = self._beam_search_and_update(next_gen, just_flushed, shared_progress, round_idx)
            else:
                self._emergency_respawn()
                just_flushed = True

            # ==========================================
            # БЛОК 5: АСИНХРОННА МІГРАЦІЯ ТА КРОСОВЕР
            # ==========================================
            curr_sims = self.ctx.sim_count
            if shared_progress is not None and (curr_sims - self._last_migration_sims) >= self._migration_interval_sims:
                self._last_migration_sims = curr_sims
                
                # Оновлюємо локальну копію глобального архіву
                if 'global_archive' in shared_progress:
                    self.global_archive = shared_progress['global_archive']
                
                # М'ЯКА МІГРАЦІЯ (Soft Migration)
                if gb is not None:
                    gb_cost, gb_sol = gb
                    
                    # Якщо воркер відстає від лідера більше ніж на 0.5%...
                    if self.run_best_cost > gb_cost * 1.005:
                        T = min(1.0, self.stagnation_counter / max(1, self.stag_limit * 3.0))
                        
                        if T < 0.3:
                            # Стан "Холодний": Експлуатація (Повне копіювання рекордсмена)
                            self.ctx.log(f" 📡 [MIGRATION] Cold state. Full copy of Global Best: {gb_cost/1e6:.4f}M$")
                            self.run_best_cost = gb_cost
                            self.run_best_sol = list(gb_sol)
                            self.pool.active_pool.insert(0, (gb_cost, gb_cost, list(gb_sol)))
                            self.stagnation_counter = 0
                            self.ipc_immunity = 50
                            
                        elif T < 0.7:
                            # Стан "Теплий": Просторовий Кросовер (Гібридизація рішень)
                            hybrid_sol = self._spatial_crossover(self.run_best_sol, gb_sol, T)
                            
                            # Оцінюємо та лікуємо створений гібрид
                            c, p, feas, _ = self.ctx.get_cached_stats(hybrid_sol)
                            if not feas or p < self.ctx.simulator.config.h_min:
                                hybrid_sol, _, _ = self.ls.heal_network(hybrid_sol, set())
                                c, p, feas, _ = self.ctx.get_cached_stats(hybrid_sol)
                                
                            if feas and c < self.run_best_cost:
                                self.ctx.log(f" 🧬 [HYBRID SUCCESS] Created new hybrid solution: {c/1e6:.4f}M$")
                                self.run_best_cost = c
                                self.run_best_sol = hybrid_sol
                                self.pool.active_pool.insert(0, (c, c, hybrid_sol))
                                self.stagnation_counter = 0
                            else:
                                self.ctx.log(f" 🧬 [HYBRID FAILED] Hybrid was too expensive. Drift continues.")
                                
                        else:
                            # Стан "Гарячий": Імунітет (Алгоритм у глибокій розвідці, ігнорує лідера)
                            self.ctx.log(f" 🛡️ [EXPLORATION SHIELD] Ignored global best. Deep Random Walk (T={T:.2f}).")

            # ==========================================
            # БЛОК 6: ЗАВЕРШЕННЯ РАУНДУ ТА ОЧИЩЕННЯ
            # ==========================================
            # Затухання нагород для Multi-Armed Bandit
            if round_idx > 0 and round_idx % 10 == 0:
                for s in self.strat_wins:
                    self.strat_wins[s] *= 0.80
                    self.strat_tries[s] = max(1.0, self.strat_tries[s] * 0.80)

            round_idx += 1
                
        # ==========================================
        # БЛОК 7: ФІНАЛІЗАЦІЯ РОБОТИ ВОРКЕРА
        # ==========================================
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
        
        # 🛡️ СТРОГИЙ ЗАХИСТ ВАЛІДНОСТІ ФІНАЛЬНОГО РЕЗУЛЬТАТУ
        # Оскільки ми використовували релаксацію штрафів, гарантуємо, 
        # що фінальний розв'язок є на 100% гідравлічно валідним.
        final_c, final_p, final_feas, _ = self.ctx.get_cached_stats(self.run_best_sol)
        if not final_feas or final_p < self.ctx.simulator.config.h_min:
            self.ctx.log(" ⚠️ [FINAL GUARD] Final solution strictly invalid. Applying hard heal.")
            healed_sol, ok, _ = self.ls.heal_network(self.run_best_sol, set())
            if ok:
                self.run_best_sol = healed_sol
                self.run_best_cost, _, _, _ = self.ctx.get_cached_stats(healed_sol)
            else:
                self.run_best_cost = float('inf') # Катастрофічний фейл (майже неможливо)
                
        return self.run_best_cost, self.run_best_sol

    # def run(self, time_budget, global_best_cost, shared_progress):
    #     self.global_best_cost = global_best_cost
    #     start_time = time.time()
    #     self.pool.clear_all()
        
    #     seeds = self.seeder.make_diverse_seeds() if self.epoch == 0 else self.seeder.make_warm_seeds(self.global_archive, self.worker_id)
    #     self.reserve_pool = self.seeder.make_reserve_pool(size=4)
        
    #     best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds]) if seeds else global_best_cost
    #     self.base_dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001 
        
    #     self._initialize_seeds(seeds)
        
    #     self.ipc_immunity = 30
        
    #     self.stag_limit = 4 
    #     round_idx = 0
    #     epoch_start_sims = self.ctx.sim_count 
    #     self._last_global_improvement_sim = self.ctx.sim_count
    #     just_flushed = False
        
    #     self._last_ping_sims = self.ctx.sim_count
    #     self._last_ping_time = time.time()
        
    #     def update_progress_ping():
    #         if shared_progress is not None:
    #             curr_sims = self.ctx.sim_count
    #             curr_time = time.time()
                
    #             if (curr_sims - self._last_ping_sims) > 10000 or (curr_time - self._last_ping_time) > 2.0:
    #                 self._last_ping_sims = curr_sims
    #                 self._last_ping_time = curr_time
                    
    #                 try:
    #                     current_data = shared_progress.get(self.worker_id)
    #                     if isinstance(current_data, dict):
    #                         new_data = dict(current_data)
    #                         new_data['sims'] = curr_sims
    #                         shared_progress[self.worker_id] = new_data
    #                 except Exception:
    #                     pass
                        
    #     self.ls.progress_callback = update_progress_ping
        
    #     while True:
    #         self.pool.current_round = round_idx 
    #         self.base_dyn_bonus = min(self.run_best_cost, self.global_best_cost) * 0.001 * random.uniform(0.70, 1.30)
            
    #         gb = shared_progress.get('global_best') if shared_progress else None
    #         if shared_progress is not None:
    #             self._process_ipc(shared_progress, gb)
    #             self._check_rescue(shared_progress, gb)
    #             shared_progress[self.worker_id] = {"round": round_idx + 1, "sims": self.ctx.sim_count, "best_cost": self.run_best_cost}
            
    #         elapsed = time.time() - start_time
    #         epoch_sims = self.ctx.sim_count - epoch_start_sims
    #         if elapsed > time_budget or epoch_sims >= self.max_sims: break
                
    #         just_flushed = self.rescue_fired
    #         self.rescue_fired = False
            
    #         self.progress_ratio = min(1.0, epoch_sims / max(1, self.max_sims)) 
    #         self.is_late_game = self.progress_ratio > 0.5 
    #         self.stag_limit = 4 + int(4 * self.progress_ratio) 
            
    #         self._check_mini_restart(shared_progress, gb)
    #         if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()
            
    #         self._apply_swap(round_idx, shared_progress)

    #         if self.stagnation_counter >= 1 or round_idx % 2 == 0:
    #             self._apply_kick(round_idx, shared_progress, gb)
                
    #         if getattr(self, '_force_flush_next', False):
    #             just_flushed = True
    #             self._force_flush_next = False

    #         next_gen = self._generate_mutations()
            
    #         if next_gen:
    #             next_gen.sort(key=lambda x: x[0])
    #             just_flushed = self._beam_search_and_update(next_gen, just_flushed, shared_progress, round_idx)
    #         else:
    #             self._emergency_respawn()
    #             just_flushed = True

    #         if round_idx > 0 and round_idx % 10 == 0:
    #             for s in self.strat_wins:
    #                 self.strat_wins[s] *= 0.80
    #                 self.strat_tries[s] = max(1.0, self.strat_tries[s] * 0.80)

    #         round_idx += 1
                
    #     if shared_progress is not None:
    #         try:
    #             current_data = shared_progress.get(self.worker_id, {})
    #             if isinstance(current_data, dict):
    #                 current_data['sims'] = self.ctx.sim_count
    #                 current_data['best_cost'] = self.run_best_cost
    #                 shared_progress[self.worker_id] = current_data
    #         except Exception:
    #             pass

    #     self.ls.progress_callback = None
        
    #     return self.run_best_cost, self.run_best_sol

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
                
        if not self.pool.active_pool:
            self.ctx.log("   > [WARNING] No valid seeds received. Injecting emergency max-diameter seed.")
            emergency = [self.ctx.max_d_idx] * self.ctx.num_pipes
            c, p, feas, _ = self.ctx.get_cached_stats(emergency)
            
            self.pool.active_pool.append((c, c, emergency))
            if feas and p >= self.ctx.simulator.config.h_min:
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
        if getattr(self, 'ipc_immunity', 0) > 0:
            self.ipc_immunity -= 1
            return
        
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
                    
                is_massively_better = peer[0] < self.run_best_cost * 0.98
                if self.stagnation_counter < self.stag_limit and not is_massively_better:
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
        
        if getattr(self, 'ipc_immunity', 0) > 0:
            return
            
        global_lag = (self.run_best_cost - self.global_best_cost) / max(self.global_best_cost, 1)
        
        if (global_lag > 0.02 and self.stagnation_counter >= self.stag_limit * 2) or global_lag > 0.05:
            if gb and gb[1]:
                self.ctx.log(f"   [RESCUE] Worker lagging by {global_lag:.1%}. Abandoning dead basin and adopting Global Best {gb[0]/1e6:.4f}M$!")
                
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
                        c, p, feas, _ = self.ctx.get_cached_stats(seed)
                        if feas and p >= self.ctx.simulator.config.h_min:
                            score = c - ((p - self.ctx.simulator.config.h_min) * self.base_dyn_bonus)
                            self.pool.active_pool.append((score, c, seed))
                            actually_added += 1
                            
                    if actually_added > 0:
                        self.stagnation_counter = 0
                        self.ctx.log(f"   [MINI-RESTART] Injected {actually_added} fresh seeds. Tabu cleared.")
                    else:
                        self.ctx.log(f"   [MINI-RESTART] All seeds infeasible. Stagnation persists.")

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
                    eff_bonus = self.base_dyn_bonus * 0.2 if p_surplus > 10.0 else self.base_dyn_bonus
                    self.pool.active_pool.insert(0, (c - (p_surplus * eff_bonus) - (self.run_best_cost * 0.1), c, swapped))

    # def _apply_kick(self, round_idx, shared_progress, gb):   
    #     n_total = sum(self.strat_tries.values())
    #     peer_archive = []
    #     if shared_progress is not None:
    #         for i in range(self.n_workers):
    #             if i != self.worker_id:
    #                 peer_data = shared_progress.get(f'best_sol_{i}')
    #                 if peer_data: peer_archive.append(peer_data)

    #     peer_to_cross = None
        
    #     if self.stagnation_counter >= self.stag_limit * 3.5:
    #         strategy = "SEGMENT_RESTART"
    #         self.pool.add_basin_to_tabu(self.run_best_sol) 
            
    #     elif self.stagnation_counter >= self.stag_limit * 3:
    #         if len(self.global_archive) >= 2:
    #             strategy = "BASIN_ESCAPE"
    #         else:
    #             strategy = "ILS_PERTURBATION"
                
    #     elif self.stagnation_counter >= self.stag_limit * 2 and peer_archive:
    #         corridor_peers = [p for p in peer_archive if 30 <= self.pool.hamming_distance(self.run_best_sol, p[1]) <= 70]
    #         diverse_peers = [p for p in peer_archive if self.pool.hamming_distance(self.run_best_sol, p[1]) > 70]
    #         if corridor_peers:
    #             strategy = "CORRIDOR_SEARCH"
    #             peer_to_cross = random.choice(corridor_peers)
    #         elif diverse_peers:
    #             if self.progress_ratio > 0.70:
    #                 strategy = "CORRIDOR_SEARCH"
    #                 peer_to_cross = random.choice(diverse_peers)
    #             else:
    #                 strategy = "IPC_CROSSOVER"
    #                 peer_to_cross = min(diverse_peers, key=lambda x: x[0])
    #         else:
    #             strategy = "ILS_PERTURBATION"
                
    #     elif self.stagnation_counter >= self.stag_limit * 1.5:
    #         if self.stagnation_counter % 3 == 0:
    #             strategy = "VNS_KICK"
    #         else:
    #             strategy = "ILS_PERTURBATION"
            
    #     elif self.stagnation_counter >= self.stag_limit:
    #         medium_kicks = ["ILS_PERTURBATION", "SUBMARINE", "VNS_KICK", "RUIN_RECREATE"]
    #         strategy = medium_kicks[self.stagnation_counter % len(medium_kicks)]
            
    #     else:
    #         valid_strats = [s for s in self.strategies if s in self.strat_wins]
    #         if not valid_strats:
    #             strategy = "ILS_PERTURBATION"
    #         else:
    #             exploration_C = 0.05
    #             strategy = max(valid_strats, key=lambda s: (self.strat_wins[s] / self.strat_tries[s]) + exploration_C * math.sqrt(math.log(max(1, n_total)) / self.strat_tries[s]))
                
    #     self.strat_tries[strategy] += 1
    #     self.ctx.log(f"[FORCE] Escalation Level: {self.stagnation_counter}/{self.stag_limit} -> Applying '{strategy}'...")
        
    #     if self.stagnation_counter >= self.stag_limit and self.pool.active_pool:
    #         valid_pool = [x for x in self.pool.active_pool if self.ctx.get_cached_stats(x[2])[1] >= self.ctx.simulator.config.h_min]
    #         source_pool = valid_pool if valid_pool else self.pool.active_pool
    #         kick_target = max([x[2] for x in source_pool], key=lambda s: self.pool.hamming_distance(s, self.run_best_sol))
    #     else:
    #         kick_mode = round_idx % 3
    #         if kick_mode == 0: kick_target = self.run_best_sol
    #         elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
    #         else:
    #             pool_sols = [x[2] for x in self.pool.active_pool[:3]]
    #             kick_target = random.choice(pool_sols) if pool_sols else self.run_best_sol
        
    #     forced_sol, locked, path_sig, failed_pipe_id = None, None, None, -1
        
    #     if not hasattr(self, 'bottleneck_failed_pipes'): self.bottleneck_failed_pipes = {}
    #     if not hasattr(self, 'loop_balance_failed_pipes'): self.loop_balance_failed_pipes = {}
        
    #     if strategy == "SEGMENT_RESTART": forced_sol, locked, log_msg = self.kicker.segment_restart_kick(kick_target, self.base_dyn_bonus)
    #     elif strategy == "IPC_CROSSOVER": forced_sol, locked, log_msg = self.kicker.crossover_with_peer_kick(kick_target, peer_to_cross[1], self.run_best_cost, peer_to_cross[0])
    #     elif strategy == "CORRIDOR_SEARCH": forced_sol, locked, log_msg = self.kicker.corridor_search_kick(kick_target, peer_to_cross[1])
    #     elif strategy == "ILS_PERTURBATION": forced_sol, locked, log_msg = self.kicker.ils_perturbation_kick(kick_target, self.stagnation_counter)
    #     elif strategy == "VNS_KICK": forced_sol, locked, log_msg = self.kicker.vns_structured_kick(kick_target, self.stagnation_counter)
    #     elif strategy == "SHOCK": forced_sol, locked, log_msg = self.kicker.forcing_hand_kick(kick_target)
    #     elif strategy == "BOTTLENECK": forced_sol, locked, log_msg, failed_pipe_id = self.kicker.upstream_bottleneck_kick(kick_target, self.bottleneck_failed_pipes, round_idx)
    #     elif strategy == "LOOP_BALANCE": 
    #         forced_sol, locked, log_msg, failed_pipe_id = self.kicker.loop_balancing_kick(kick_target, self.base_dyn_bonus, self.loop_balance_failed_pipes, round_idx)
    #     elif strategy == "DIAM_DIVERSITY": forced_sol, locked, log_msg = self.kicker.diameter_diversity_kick(kick_target, self.stagnation_counter) 
    #     elif strategy == "RUIN_RECREATE": forced_sol, locked, log_msg = self.kicker.ruin_and_recreate_kick(kick_target, self.stagnation_counter) 
    #     elif strategy == "FINISHER": forced_sol, locked, log_msg = self.kicker.micro_trim_kick(kick_target)
    #     elif strategy == "SYNC_TRIM": forced_sol, locked, log_msg = self.kicker.sync_trim_kick(kick_target, self.base_dyn_bonus)
    #     elif strategy == "ZERO_SUM": forced_sol, locked, log_msg = self.kicker.zero_sum_shift_kick(kick_target)
    #     elif strategy == "SUBMARINE": forced_sol, locked, log_msg = self.kicker.submarine_oscillation_kick(kick_target)
    #     elif strategy == "BASIN_ESCAPE": forced_sol, locked, log_msg = self.kicker.basin_escape(kick_target, self.global_archive)
    #     else: forced_sol, locked, log_msg, path_sig = self.kicker.topological_inversion_kick(kick_target, self.pool.kick_tabu_set)

    #     if forced_sol is None or locked is None:
    #         if failed_pipe_id != -1: 
    #             if strategy == "BOTTLENECK": self.bottleneck_failed_pipes[failed_pipe_id] = round_idx
    #             elif strategy == "LOOP_BALANCE": self.loop_balance_failed_pipes[failed_pipe_id] = round_idx
    #         reason = log_msg if log_msg else "Unhealable structural damage"
    #         self.ctx.log(f"      -> Kick '{strategy}' failed. Reason: {reason}")
    #         return
            
    #     if failed_pipe_id != -1: 
    #         if strategy == "BOTTLENECK": self.bottleneck_failed_pipes[failed_pipe_id] = round_idx
    #         elif strategy == "LOOP_BALANCE": self.loop_balance_failed_pipes[failed_pipe_id] = round_idx

    #     c, p, feas, _ = self.ctx.get_cached_stats(forced_sol)
    #     if not feas or p < self.ctx.simulator.config.h_min:
    #         forced_sol, ok, _ = self.ls.heal_network(forced_sol, locked)
    #         if not ok:
    #             if log_msg: self.ctx.log(f"     -> {log_msg}")
    #             self.ctx.log(f"     -> Kick Failed (Unhealable structural damage). Discarded.")
    #             return

    #     if log_msg: self.ctx.log(f"     -> {log_msg}")
            
    #     base_margin = 0.04 * (1.0 - self.progress_ratio) + 0.015
    #     stag_factor = max(1.0, self.stagnation_counter / max(1, self.stag_limit))
        
    #     dynamic_margin = base_margin * stag_factor
    #     water_level = self.run_best_cost * (1.0 + dynamic_margin)
        
    #     if strategy == "SEGMENT_RESTART":
    #         final_sol = forced_sol 
    #         self.pool.basin_tabu.clear() 
    #     else:
    #         is_heavy = strategy in ["ILS_PERTURBATION", "VNS_KICK", "RUIN_RECREATE", "IPC_CROSSOVER", "CORRIDOR_SEARCH", "SUBMARINE", "BASIN_ESCAPE"]
    #         quick_passes = 1
    #         locked_for_squeeze = set() if strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "CORRIDOR_SEARCH"] else locked
            
    #         quick_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked_for_squeeze, max_passes=quick_passes, quick_mode=True, dyn_bonus=self.base_dyn_bonus)
    #         quick_cost, quick_p, quick_f, _ = self.ctx.get_cached_stats(quick_sol)
            
    #         hb_margin = 0.03 - (0.02 * self.progress_ratio)
    #         hyperband_threshold = self.run_best_cost * (1.0 + hb_margin)
            
    #         is_promising = quick_f and quick_p >= self.ctx.simulator.config.h_min and (quick_cost < hyperband_threshold)
            
    #         if is_promising:
    #             gap = (quick_cost - self.run_best_cost) / max(self.run_best_cost, 1.0)
                
    #             if gap < -0.0001: 
    #                 deep_passes = 10 if self.ctx.num_pipes >= 200 else 6 
    #             elif gap <= 0.002:  
    #                 deep_passes = 4 
    #             elif gap <= 0.02:   
    #                 deep_passes = 3
    #             elif gap <= 0.05:
    #                 deep_passes = 2
    #             else:              
    #                 deep_passes = 1
                
    #             consensus_locked = set()
    #             if len(self.global_archive) >= 3 and self.is_late_game and self.stagnation_counter < 2 and not getattr(self, '_force_flush_next', False):
    #                 arch_sols = [x[1] for x in self.global_archive]
                    
    #                 raw_consensus = set()
    #                 for i in range(self.ctx.num_pipes):
    #                     if all(sol[i] == arch_sols[0][i] for sol in arch_sols):
    #                         raw_consensus.add(i)
                    
    #                 freeze_pct = 0.25 if self.progress_ratio > 0.6 else 0.10
    #                 max_frozen = max(5, int(self.ctx.num_pipes * freeze_pct))
                    
    #                 if self.progress_ratio > 0.88:
    #                     max_frozen = 0
                    
    #                 if len(raw_consensus) > max_frozen and max_frozen > 0:
    #                     interesting = [i for i in raw_consensus if 0 < arch_sols[0][i] < self.ctx.max_d_idx]
    #                     consensus_locked = set(list(interesting)[:max_frozen])
    #                 elif max_frozen > 0:
    #                     consensus_locked = raw_consensus
    #                 else:
    #                     consensus_locked = set()
                            
    #             safe_consensus = consensus_locked - (locked if locked else set())
    #             final_locked = locked_for_squeeze.union(safe_consensus)
                
    #             self.ctx.log(f"        [HYPERBAND] Gap {gap:.1%}. Deep Squeeze ({deep_passes} passes, {len(safe_consensus)} frozen)...")
                
    #             final_sol = self.ls.gradient_squeeze(quick_sol, locked_pipes=final_locked, max_passes=deep_passes, dyn_bonus=self.base_dyn_bonus)
    #         else:
    #             final_sol = quick_sol
        
    #     c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
    #     if feas and c > self.run_best_cost * 1.5:
    #         self.ctx.log(f"     -> Hard-Rejected (Cost Explosion): {c/1e6:.4f}M$")
    #         feas = False
        
    #     if feas and p >= self.ctx.simulator.config.h_min:
    #         p_surplus = p - self.ctx.simulator.config.h_min
            
    #         if p_surplus > 10.0:
    #             eff_bonus = self.base_dyn_bonus * 0.2 
    #         else:
    #             eff_bonus = self.base_dyn_bonus
            
    #         if p_surplus < 2.0:
    #             pre_sq_sol = final_sol 
    #             final_sol = self.ls.gradient_squeeze(final_sol, locked_pipes=set(), max_passes=3, quick_mode=True, dyn_bonus=eff_bonus)
    #             c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                
    #             if not feas or p < self.ctx.simulator.config.h_min:
    #                 self.ctx.log(f"     -> Post-squeeze became infeasible. Reverting to pre-squeeze state.")
    #                 final_sol = pre_sq_sol
    #                 c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                    
    #             p_surplus = p - self.ctx.simulator.config.h_min if feas else p_surplus
    #             eff_bonus = self.base_dyn_bonus * 0.3 if p_surplus < 1.0 else self.base_dyn_bonus
            
    #         score = c - (p_surplus * eff_bonus)
            
    #         if c < self.run_best_cost:
    #             is_ghost = False
    #             if (self.run_best_cost - c) > (self.run_best_cost * 0.02):
    #                 is_ghost = self.ctx.is_ghost_solution(final_sol, c)
                    
    #             if is_ghost:
    #                 self.ctx.log(f"   > [SHIELD] Force illusion blocked ({c/1e6:.4f}M$)!")
    #             else:
    #                 diff = self.run_best_cost - c
    #                 self.run_best_cost, self.run_best_sol = c, final_sol
    #                 self.pool.active_pool.insert(0, (score, c, final_sol))
                    
    #                 if diff > (self.run_best_cost * 0.001):
    #                     self.stagnation_counter = 0 
    #                 else:
    #                     self.stagnation_counter = max(0, self.stagnation_counter - 2)
                    
    #                 improvement_pct = diff / self.run_best_cost
    #                 reward = 5.0 if improvement_pct > 0.01 else (3.0 if improvement_pct > 0.001 else 1.0)
                    
    #                 self.strat_wins[strategy] += reward
    #                 self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
    #                 self._update_global_best(shared_progress)

    #         elif strategy in ["SEGMENT_RESTART", "BASIN_ESCAPE"]:
    #             self.pool.active_pool.clear() 
    #             self.pool.active_pool.append((score, c, final_sol))
    #             self.stagnation_counter = 0 
    #             self.corridor_pool_streak = 0
    #             self._force_flush_next = True 
                
    #             self.run_best_cost = c
    #             self.run_best_sol = list(final_sol)
    #             self.ipc_immunity = 50
                
    #             if c < self.global_best_cost:
    #                 self._update_global_best(shared_progress)
                
    #             if strategy == "SEGMENT_RESTART": 
    #                 self.pool.basin_tabu.clear() 
    #             self.ctx.log(f"      -> POOL FLUSHED. Adopted Major Escape: {c/1e6:.4f}M$")
                
    #         elif c < water_level and not self.pool.is_basin_tabu(final_sol):
    #             pool_gap = (c - self.run_best_cost) / max(self.run_best_cost, 1)
    #             max_pool_gap = 0.06 if self.is_late_game else 0.12
                
    #             if self.ctx.num_pipes >= 200 and pool_gap > max_pool_gap:
    #                 self.ctx.log(f"     -> Pool Filtered (gap {pool_gap:.1%}): {c/1e6:.4f}M$")
    #             else:
    #                 self.pool.active_pool.append((score * 1.05, c, final_sol))
    #                 if strategy == "CORRIDOR_SEARCH":
    #                     self.corridor_pool_streak += 1
    #                     if self.corridor_pool_streak <= 3: self.stagnation_counter = max(0, self.stagnation_counter - 2) 
    #                 elif strategy in ["ILS_PERTURBATION", "IPC_CROSSOVER", "RUIN_RECREATE", "VNS_KICK", "ZERO_SUM", "SUBMARINE"]:
    #                     self.corridor_pool_streak = 0
    #                 elif strategy == "BASIN_ESCAPE":
    #                     self.corridor_pool_streak = 0
    #                     self.stagnation_counter = max(0, self.stagnation_counter - int(self.stag_limit * 1.5))
                        
    #                 self.strat_wins[strategy] += 0.5
    #                 self.ctx.log(f"      -> Added to Pool (Water Level Accept): {c/1e6:.4f}M$")
                    
    #         elif self.stagnation_counter >= self.stag_limit * 3.5 and not self.pool.is_basin_tabu(final_sol):
    #             self.pool.active_pool.append((score * 1.20, c, final_sol))
    #             self.stagnation_counter = int(self.stag_limit * 0.5) 
    #             self.ctx.log(f"     -> 🚀 HAIL MARY ACCEPT (Forced Escape): {c/1e6:.4f}M$")
    #         else:
    #             self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {c/1e6:.4f}M$")
    #         if path_sig: self.pool.kick_tabu_set.add(path_sig)
    #     else:
    #          self.ctx.log(f"     -> Injection/Squeeze Failed: Infeasible/Exploded")
    
    def _apply_kick(self, round_idx, shared_progress, gb):
        n_total = sum(self.strat_tries.values())

        # ==========================================
        # 1. ОБЧИСЛЕННЯ ТЕМПЕРАТУРИ (T)
        # ==========================================
        # T зростає від 0.0 (знайшли рекорд) до 1.0 (глухий кут)
        T = min(1.0, self.stagnation_counter / max(1, self.stag_limit * 3.0))

        # ==========================================
        # 2. ВІДБІР СТРАТЕГІЙ ЗАЛЕЖНО ВІД T
        # ==========================================
        if T >= 0.9:
            pool_strats = ["BASIN_ESCAPE", "SPATIAL_PERTURB", "RUIN_RECREATE"]
            if len(self.global_archive) < 2: pool_strats.remove("BASIN_ESCAPE")
        elif T >= 0.5:
            pool_strats = ["SPATIAL_PERTURB", "RUIN_RECREATE", "TOPO_INV", "ZERO_SUM"]
        else:
            pool_strats = ["SHOCK", "BOTTLENECK", "LOOP_BALANCE", "ZERO_SUM", "TRIM"]

        # Відсікаємо LOOP_BALANCE для дерев
        if nx.is_tree(self.ctx.base_G_flow) and "LOOP_BALANCE" in pool_strats:
            pool_strats.remove("LOOP_BALANCE")

        valid_strats = [s for s in pool_strats if s in self.strat_wins]
        untried_strats = [s for s in pool_strats if s not in self.strat_wins]
        
        if untried_strats: 
            strategy = random.choice(untried_strats)
        else:
            exploration_C = 0.05 + (0.1 * T) 
            strategy = max(valid_strats, key=lambda s: (self.strat_wins[s] / max(1, self.strat_tries.get(s, 1))) + exploration_C * math.sqrt(math.log(max(1, n_total)) / max(1, self.strat_tries.get(s, 1))))

        self.strat_tries[strategy] = self.strat_tries.get(strategy, 0) + 1
        self.ctx.log(f"[FORCE] Temp: {T:.2f} (Stag: {self.stagnation_counter}/{self.stag_limit}) -> Applying '{strategy}'...")

        # ==========================================
        # 3. ВИБІР ЦІЛІ (Target Solution)
        # ==========================================
        if T >= 0.8 and self.pool.active_pool:
            # АНТИ-ЕЛІТИЗМ
            source_pool = [x for x in self.pool.active_pool if self.ctx.get_cached_stats(x[2])[1] >= self.ctx.simulator.config.h_min]
            if not source_pool: source_pool = self.pool.active_pool
            kick_target = max([x[2] for x in source_pool], key=lambda s: self.pool.hamming_distance(s, self.run_best_sol))
            
        elif T >= 0.4 and self.pool.active_pool:
            # ВИПАДКОВЕ БЛУКАННЯ
            kick_target = random.choice([x[2] for x in self.pool.active_pool])
            
        else:
            # Експлуатація 
            kick_mode = round_idx % 3
            if kick_mode == 0: kick_target = self.run_best_sol
            elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
            else: kick_target = self.run_best_sol

        forced_sol, locked, path_sig, failed_pipe_id = None, None, None, -1
        
        if not hasattr(self, 'bottleneck_failed_pipes'): self.bottleneck_failed_pipes = {}
        if not hasattr(self, 'loop_balance_failed_pipes'): self.loop_balance_failed_pipes = {}

        # ==========================================
        # 4. ДИНАМІЧНИЙ ВИКЛИК (Передаємо T)
        # ==========================================
        kick_args = {
            'T': T,
            'failed_pipes': self.bottleneck_failed_pipes if strategy == "BOTTLENECK" else self.loop_balance_failed_pipes,
            'current_round': round_idx,
            'tabu_set': self.pool.kick_tabu_set,
            'dyn_bonus': self.base_dyn_bonus,
            'global_archive': self.global_archive
        }

        try:
            if strategy == "SHOCK": res = self.kicker.forcing_hand_kick(kick_target, **kick_args)
            elif strategy == "BOTTLENECK": res = self.kicker.upstream_bottleneck_kick(kick_target, **kick_args)
            elif strategy == "TOPO_INV": res = self.kicker.topological_inversion_kick(kick_target, **kick_args)
            elif strategy == "LOOP_BALANCE": res = self.kicker.loop_balancing_kick(kick_target, **kick_args)
            elif strategy == "ZERO_SUM": res = self.kicker.zero_sum_shift_kick(kick_target, **kick_args)
            elif strategy == "SPATIAL_PERTURB": res = self.kicker.spatial_perturb_kick(kick_target, **kick_args)
            elif strategy == "TRIM": res = self.kicker.peripheral_trim_kick(kick_target, **kick_args)
            elif strategy == "SMART_PERTURB": res = self.kicker.smart_perturbation_kick(kick_target, **kick_args)
            elif strategy == "RUIN_RECREATE": res = self.kicker.ruin_and_recreate_kick(kick_target, **kick_args)
            elif strategy == "BASIN_ESCAPE": res = self.kicker.basin_escape(kick_target, **kick_args)
            else: res = self.kicker.forcing_hand_kick(kick_target, **kick_args)

            if len(res) == 4:
                forced_sol, locked, log_msg, extra_info = res
                if strategy in ["BOTTLENECK", "LOOP_BALANCE"]: failed_pipe_id = extra_info
                elif strategy == "TOPO_INV": path_sig = extra_info
            else:
                forced_sol, locked, log_msg = res

        except Exception as e:
            self.ctx.log(f"      -> Critical execution error in {strategy}: {e}")
            return

        # ==========================================
        # 5. ОБРОБКА ПОМИЛОК КІКА
        # ==========================================
        if forced_sol is None or locked is None:
            if failed_pipe_id != -1: 
                if strategy == "BOTTLENECK": self.bottleneck_failed_pipes[failed_pipe_id] = round_idx
                elif strategy == "LOOP_BALANCE": self.loop_balance_failed_pipes[failed_pipe_id] = round_idx
            reason = log_msg if log_msg else "Unhealable structural damage / No targets"
            self.ctx.log(f"      -> Kick '{strategy}' failed. Reason: {reason}")
            return
            
        if failed_pipe_id != -1: 
            if strategy == "BOTTLENECK": self.bottleneck_failed_pipes[failed_pipe_id] = round_idx
            elif strategy == "LOOP_BALANCE": self.loop_balance_failed_pipes[failed_pipe_id] = round_idx

        # ==========================================
        # 6. ХІЛІНГ З РЕЛАКСАЦІЄЮ ШТРАФІВ
        # ==========================================
        c, p, feas, _ = self.ctx.get_cached_stats(forced_sol)
        
        max_allowed_deficit = 0.5 * T
        relaxed_h_min = self.ctx.simulator.config.h_min - max_allowed_deficit
        
        if not feas or p < relaxed_h_min:
            # 🔴 ВИПРАВЛЕННЯ: Знімаємо кайдани з хілера для масових мутацій
            heal_locks = set() if strategy in ["SPATIAL_PERTURB", "SMART_PERTURB", "RUIN_RECREATE"] else locked
            
            forced_sol, ok, _ = self.ls.heal_network(forced_sol, heal_locks)
            if not ok:
                if log_msg: self.ctx.log(f"     -> {log_msg}")
                self.ctx.log(f"     -> Kick Failed (Unhealable structural damage). Discarded.")
                return

        if log_msg: self.ctx.log(f"     -> {log_msg}")
            
        base_margin = 0.04 * (1.0 - self.progress_ratio) + 0.015
        dynamic_margin = base_margin * (1.0 + T) 
        water_level = self.run_best_cost * (1.0 + dynamic_margin)
        
        # ==========================================
        # 7. ГЛИБОКЕ СКВІЗУВАННЯ
        # ==========================================
        if strategy == "BASIN_ESCAPE":
            final_sol = forced_sol 
            if self.run_best_sol:
                self.pool.basin_tabu.add(tuple(self.run_best_sol))
                
            if len(self.pool.basin_tabu) > 100:
                self.pool.basin_tabu.pop()
        elif T >= 0.7:
            final_sol = forced_sol
            self.ctx.log(f"        [EXPLORATION] Squeeze bypassed (T={T:.2f}). Letting solution drift.")
        else:
            quick_passes = 1 if T < 0.4 else 0
            
            locked_for_squeeze = set() if strategy in ["SMART_PERTURB", "SPATIAL_PERTURB", "RUIN_RECREATE"] else locked
            
            if quick_passes > 0:
                quick_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked_for_squeeze, max_passes=quick_passes, quick_mode=True, dyn_bonus=self.base_dyn_bonus)
            else:
                quick_sol = forced_sol
                
            quick_cost, quick_p, quick_f, _ = self.ctx.get_cached_stats(quick_sol)
            
            hb_margin = 0.03 - (0.02 * self.progress_ratio)
            hyperband_threshold = self.run_best_cost * (1.0 + hb_margin)
            
            is_promising = quick_f and quick_p >= self.ctx.simulator.config.h_min and (quick_cost < hyperband_threshold)
            
            if is_promising:
                gap = (quick_cost - self.run_best_cost) / max(self.run_best_cost, 1.0)
                
                if T >= 0.4: deep_passes = 1
                elif gap < -0.0001: deep_passes = 10 if self.ctx.num_pipes >= 200 else 6 
                elif gap <= 0.002: deep_passes = 4 
                elif gap <= 0.02: deep_passes = 3
                elif gap <= 0.05: deep_passes = 2
                else: deep_passes = 1
                
                consensus_locked = set()
                if len(self.global_archive) >= 3 and self.is_late_game and T < 0.2 and not getattr(self, '_force_flush_next', False):
                    arch_sols = [x[1] for x in self.global_archive]
                    raw_consensus = set()
                    for i in range(self.ctx.num_pipes):
                        if all(sol[i] == arch_sols[0][i] for sol in arch_sols):
                            raw_consensus.add(i)
                    
                    freeze_pct = 0.25 if self.progress_ratio > 0.6 else 0.10
                    max_frozen = max(5, int(self.ctx.num_pipes * freeze_pct))
                    if self.progress_ratio > 0.88: max_frozen = 0
                    
                    if len(raw_consensus) > max_frozen and max_frozen > 0:
                        interesting = [i for i in raw_consensus if 0 < arch_sols[0][i] < self.ctx.max_d_idx]
                        consensus_locked = set(list(interesting)[:max_frozen])
                    elif max_frozen > 0:
                        consensus_locked = raw_consensus
                            
                safe_consensus = consensus_locked - (locked if locked else set())
                final_locked = locked_for_squeeze.union(safe_consensus)
                
                self.ctx.log(f"        [HYPERBAND] Gap {gap:.1%}. Deep Squeeze ({deep_passes} passes, {len(safe_consensus)} frozen)...")
                final_sol = self.ls.gradient_squeeze(quick_sol, locked_pipes=final_locked, max_passes=deep_passes, dyn_bonus=self.base_dyn_bonus)
            else:
                final_sol = quick_sol

        # ==========================================
        # 8. ОЦІНКА ШТРАФУ ТА ВАРТОСТІ
        # ==========================================
        c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
        
        # Обчислюємо дефіцит тиску (наскільки не дотягуємо до h_min)
        deficit = max(0.0, self.ctx.simulator.config.h_min - p) if feas else float('inf')
        is_relaxed_valid = feas and (deficit <= max_allowed_deficit)

        if is_relaxed_valid:
            # 🔴 ШТРАФ: 200,000$ за кожен метр порушення тиску
            penalty = deficit * 200000.0
            effective_cost = c + penalty
            
            if effective_cost > self.run_best_cost * 1.5:
                self.ctx.log(f"     -> Hard-Rejected (Cost Explosion): {effective_cost/1e6:.4f}M$")
                is_relaxed_valid = False
                
        if is_relaxed_valid:
            # Розрахунок Score (для сортування в пулі)
            if deficit == 0:
                p_surplus = max(0.0, p - self.ctx.simulator.config.h_min)
                eff_bonus = self.base_dyn_bonus * 0.2 if p_surplus > 10.0 else self.base_dyn_bonus
                score = effective_cost - (p_surplus * eff_bonus)
            else:
                score = effective_cost # Для невалідних рішень бонусів немає

            # ==========================================
            # 9. ДИНАМІЧНА НАГОРОДА ТА ОНОВЛЕННЯ РЕКОРДІВ
            # ==========================================
            # 🛡️ СТРОГИЙ ЗАХИСТ: Рекорди оновлюються ТІЛЬКИ якщо рішення ідеально валідне (deficit == 0)
            if deficit == 0 and effective_cost < self.run_best_cost:
                is_ghost = False
                if (self.run_best_cost - effective_cost) > (self.run_best_cost * 0.02):
                    is_ghost = self.ctx.is_ghost_solution(final_sol, effective_cost)
                    
                if is_ghost:
                    self.ctx.log(f"   > [SHIELD] Force illusion blocked ({effective_cost/1e6:.4f}M$)!")
                else:
                    diff = self.run_best_cost - effective_cost
                    self.run_best_cost, self.run_best_sol = effective_cost, final_sol
                    self.pool.active_pool.insert(0, (score, effective_cost, final_sol))
                    
                    if diff > (self.run_best_cost * 0.001): self.stagnation_counter = 0 
                    else: self.stagnation_counter = max(0, self.stagnation_counter - 2)
                    
                    improvement_pct = diff / self.run_best_cost
                    reward = 10.0 * improvement_pct * 100 
                    self.strat_wins[strategy] = self.strat_wins.get(strategy, 0) + max(1.0, reward)
                    
                    self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
                    self._update_global_best(shared_progress)

            elif strategy == "BASIN_ESCAPE" and deficit == 0:
                self.pool.active_pool.clear() 
                self.pool.active_pool.append((score, effective_cost, final_sol))
                self.stagnation_counter = 0 
                self.corridor_pool_streak = 0
                self._force_flush_next = True 
                
                self.run_best_cost = effective_cost
                self.run_best_sol = list(final_sol)
                self.ipc_immunity = 50
                
                if effective_cost < self.global_best_cost: self._update_global_best(shared_progress)
                self.ctx.log(f"      -> POOL FLUSHED. Adopted Major Escape: {effective_cost/1e6:.4f}M$")
                
            elif effective_cost < water_level and not self.pool.is_basin_tabu(final_sol):
                hamming_dist = self.pool.hamming_distance(final_sol, self.run_best_sol)
                diversity_ratio = hamming_dist / self.ctx.num_pipes
                
                if T >= 0.5 and diversity_ratio > 0.05:
                    reward = 2.0 * diversity_ratio * T
                    self.strat_wins[strategy] = self.strat_wins.get(strategy, 0) + reward

                pool_gap = (effective_cost - self.run_best_cost) / max(self.run_best_cost, 1)
                max_pool_gap = 0.06 if self.is_late_game else 0.12
                
                if self.ctx.num_pipes >= 200 and pool_gap > max_pool_gap:
                    self.ctx.log(f"     -> Pool Filtered (gap {pool_gap:.1%}): {effective_cost/1e6:.4f}M$")
                else:
                    self.pool.active_pool.append((score * 1.05, effective_cost, final_sol))
                    if strategy in ["SMART_PERTURB", "RUIN_RECREATE", "ZERO_SUM", "LOOP_BALANCE"]:
                        self.corridor_pool_streak = 0
                        
                    # Сповіщення в лог про те, що невалідне рішення пустили в пул
                    if deficit > 0:
                        self.ctx.log(f"      ⚠️ [RELAXATION] Pooled invalid sol (p={p:.2f}m, eff_cost={effective_cost/1e6:.4f}M$)")
                    else:
                        self.ctx.log(f"      -> Added to Pool (Water Level Accept): {effective_cost/1e6:.4f}M$")
                    
            elif T >= 0.95 and not self.pool.is_basin_tabu(final_sol):
                self.pool.active_pool.append((score * 1.20, effective_cost, final_sol))
                self.stagnation_counter = int(self.stag_limit * 0.5) 
                self.ctx.log(f"     -> 🚀 HAIL MARY ACCEPT (Forced Escape): {effective_cost/1e6:.4f}M$")
            else:
                self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {effective_cost/1e6:.4f}M$")
            
            if path_sig: self.pool.kick_tabu_set.add(path_sig)
        else:
             self.ctx.log(f"     -> Injection/Squeeze Failed: Infeasible/Exploded")

        # ЗОЛОТЕ ПРАВИЛО: "Забування" минулих нагород (Decay). 
        for k in list(self.strat_wins.keys()):
            self.strat_wins[k] *= 0.95
            self.strat_tries[k] = max(1.0, self.strat_tries[k] * 0.95)
    
    # def _apply_kick(self, round_idx, shared_progress, gb):
    #     n_total = sum(self.strat_tries.values())

    #     # ==========================================
    #     # 1. ОБЧИСЛЕННЯ ТЕМПЕРАТУРИ (T)
    #     # ==========================================
    #     # T зростає від 0.0 (знайшли рекорд) до 1.0 (глухий кут)
    #     T = min(1.0, self.stagnation_counter / max(1, self.stag_limit * 3.0))

    #     # ==========================================
    #     # 2. ВІДБІР СТРАТЕГІЙ ЗАЛЕЖНО ВІД T
    #     # ==========================================
    #     if T >= 0.9:
    #         pool_strats = ["BASIN_ESCAPE", "SPATIAL_PERTURB", "RUIN_RECREATE"]
    #         if len(self.global_archive) < 2: pool_strats.remove("BASIN_ESCAPE")
    #     elif T >= 0.5:
    #         pool_strats = ["SPATIAL_PERTURB", "RUIN_RECREATE", "TOPO_INV", "ZERO_SUM"]
    #     else:
    #         pool_strats = ["SHOCK", "BOTTLENECK", "LOOP_BALANCE", "ZERO_SUM", "TRIM"]

    #     # Відсікаємо LOOP_BALANCE для дерев
    #     if nx.is_tree(self.ctx.base_G_flow) and "LOOP_BALANCE" in pool_strats:
    #         pool_strats.remove("LOOP_BALANCE")

    #     # Multi-Armed Bandit для вибору найкращої стратегії з пулу
    #     valid_strats = [s for s in pool_strats if s in self.strat_wins]
    #     if not valid_strats: 
    #         strategy = random.choice(pool_strats)
    #     else:
    #         exploration_C = 0.05 + (0.1 * T) # Більше дослідження при високій T
    #         strategy = max(valid_strats, key=lambda s: (self.strat_wins[s] / max(1, self.strat_tries.get(s, 1))) + exploration_C * math.sqrt(math.log(max(1, n_total)) / max(1, self.strat_tries.get(s, 1))))

    #     self.strat_tries[strategy] = self.strat_tries.get(strategy, 0) + 1
    #     self.ctx.log(f"[FORCE] Temp: {T:.2f} (Stag: {self.stagnation_counter}/{self.stag_limit}) -> Applying '{strategy}'...")

    #     # ==========================================
    #     # 3. ВИБІР ЦІЛІ (Target Solution)
    #     # ==========================================
    #     # Чим вища T, тим частіше беремо віддалені рішення з пулу
    #     if T >= 0.8 and self.pool.active_pool:
    #         # 🔴 АНТИ-ЕЛІТИЗМ: При дуже високій T беремо найвіддаленіше рішення, 
    #         # щоб стартувати стрибок якомога далі від гравітаційної ями чемпіона
    #         source_pool = [x for x in self.pool.active_pool if self.ctx.get_cached_stats(x[2])[1] >= self.ctx.simulator.config.h_min]
    #         if not source_pool: source_pool = self.pool.active_pool
    #         kick_target = max([x[2] for x in source_pool], key=lambda s: self.pool.hamming_distance(s, self.run_best_sol))
            
    #     elif T >= 0.4 and self.pool.active_pool:
    #         # 🔴 ВИПАДКОВЕ БЛУКАННЯ: При середній T даємо шанс будь-якому рішенню в пулі
    #         kick_target = random.choice([x[2] for x in self.pool.active_pool])
            
    #     else:
    #         # Експлуатація (Холодний алгоритм шліфує найкращі рішення)
    #         kick_mode = round_idx % 3
    #         if kick_mode == 0: kick_target = self.run_best_sol
    #         elif kick_mode == 1 and self.pool.active_pool: kick_target = self.pool.active_pool[0][2]
    #         else: kick_target = self.run_best_sol

    #     forced_sol, locked, path_sig, failed_pipe_id = None, None, None, -1
        
    #     if not hasattr(self, 'bottleneck_failed_pipes'): self.bottleneck_failed_pipes = {}
    #     if not hasattr(self, 'loop_balance_failed_pipes'): self.loop_balance_failed_pipes = {}

    #     # ==========================================
    #     # 4. ДИНАМІЧНИЙ ВИКЛИК (Передаємо T)
    #     # ==========================================
    #     kick_args = {
    #         'T': T,
    #         'failed_pipes': self.bottleneck_failed_pipes if strategy == "BOTTLENECK" else self.loop_balance_failed_pipes,
    #         'current_round': round_idx,
    #         'tabu_set': self.pool.kick_tabu_set,
    #         'dyn_bonus': self.base_dyn_bonus,
    #         'global_archive': self.global_archive
    #     }

    #     try:
    #         if strategy == "SHOCK": res = self.kicker.forcing_hand_kick(kick_target, **kick_args)
    #         elif strategy == "BOTTLENECK": res = self.kicker.upstream_bottleneck_kick(kick_target, **kick_args)
    #         elif strategy == "TOPO_INV": res = self.kicker.topological_inversion_kick(kick_target, **kick_args)
    #         elif strategy == "LOOP_BALANCE": res = self.kicker.loop_balancing_kick(kick_target, **kick_args)
    #         elif strategy == "ZERO_SUM": res = self.kicker.zero_sum_shift_kick(kick_target, **kick_args)
    #         elif strategy == "SPATIAL_PERTURB": res = self.kicker.spatial_perturb_kick(kick_target, **kick_args)
    #         elif strategy == "TRIM": res = self.kicker.peripheral_trim_kick(kick_target, **kick_args)
    #         elif strategy == "SMART_PERTURB": res = self.kicker.smart_perturbation_kick(kick_target, **kick_args)
    #         elif strategy == "RUIN_RECREATE": res = self.kicker.ruin_and_recreate_kick(kick_target, **kick_args)
    #         elif strategy == "BASIN_ESCAPE": res = self.kicker.basin_escape(kick_target, **kick_args)
    #         else: res = self.kicker.forcing_hand_kick(kick_target, **kick_args)

    #         # Розпаковка результатів
    #         if len(res) == 4:
    #             forced_sol, locked, log_msg, extra_info = res
    #             if strategy in ["BOTTLENECK", "LOOP_BALANCE"]: failed_pipe_id = extra_info
    #             elif strategy == "TOPO_INV": path_sig = extra_info
    #         else:
    #             forced_sol, locked, log_msg = res

    #     except Exception as e:
    #         self.ctx.log(f"      -> Critical execution error in {strategy}: {e}")
    #         return

    #     # ==========================================
    #     # 5. ОБРОБКА ПОМИЛОК КІКА
    #     # ==========================================
    #     if forced_sol is None or locked is None:
    #         if failed_pipe_id != -1: 
    #             if strategy == "BOTTLENECK": self.bottleneck_failed_pipes[failed_pipe_id] = round_idx
    #             elif strategy == "LOOP_BALANCE": self.loop_balance_failed_pipes[failed_pipe_id] = round_idx
    #         reason = log_msg if log_msg else "Unhealable structural damage / No targets"
    #         self.ctx.log(f"      -> Kick '{strategy}' failed. Reason: {reason}")
    #         return
            
    #     if failed_pipe_id != -1: 
    #         if strategy == "BOTTLENECK": self.bottleneck_failed_pipes[failed_pipe_id] = round_idx
    #         elif strategy == "LOOP_BALANCE": self.loop_balance_failed_pipes[failed_pipe_id] = round_idx

    #     # ==========================================
    #     # 6. ХІЛІНГ (ЗЦІЛЕННЯ) ТА ОЦІНКА ВАРТОСТІ
    #     # ==========================================
    #     c, p, feas, _ = self.ctx.get_cached_stats(forced_sol)
    #     if not feas or p < self.ctx.simulator.config.h_min:
    #         forced_sol, ok, _ = self.ls.heal_network(forced_sol, locked)
    #         if not ok:
    #             if log_msg: self.ctx.log(f"     -> {log_msg}")
    #             self.ctx.log(f"     -> Kick Failed (Unhealable structural damage). Discarded.")
    #             return

    #     if log_msg: self.ctx.log(f"     -> {log_msg}")
            
    #     base_margin = 0.04 * (1.0 - self.progress_ratio) + 0.015
    #     dynamic_margin = base_margin * (1.0 + T) # Допуск до басейну зростає з температурою
    #     water_level = self.run_best_cost * (1.0 + dynamic_margin)
        
    #     # ==========================================
    #     # 7. ГЛИБОКЕ СКВІЗУВАННЯ (АБСОЛЮТНА АНТИ-ГРАВІТАЦІЯ)
    #     # ==========================================
    #     if strategy == "BASIN_ESCAPE":
    #         final_sol = forced_sol 
    #         self.pool.basin_tabu.clear() 
    #     elif T >= 0.7:
    #         # 🔴 Якщо нам гаряче - НІЯКОГО сквізування. Взагалі. Дозволяємо мутанту жити.
    #         final_sol = forced_sol
    #         self.ctx.log(f"        [EXPLORATION] Squeeze bypassed (T={T:.2f}). Letting solution drift.")
    #     else:
    #         quick_passes = 1 if T < 0.4 else 0
    #         locked_for_squeeze = set() if strategy in ["SMART_PERTURB"] else locked
            
    #         if quick_passes > 0:
    #             quick_sol = self.ls.gradient_squeeze(forced_sol, locked_pipes=locked_for_squeeze, max_passes=quick_passes, quick_mode=True, dyn_bonus=self.base_dyn_bonus)
    #         else:
    #             quick_sol = forced_sol
                
    #         quick_cost, quick_p, quick_f, _ = self.ctx.get_cached_stats(quick_sol)
            
    #         hb_margin = 0.03 - (0.02 * self.progress_ratio)
    #         hyperband_threshold = self.run_best_cost * (1.0 + hb_margin)
            
    #         is_promising = quick_f and quick_p >= self.ctx.simulator.config.h_min and (quick_cost < hyperband_threshold)
            
    #         if is_promising:
    #             gap = (quick_cost - self.run_best_cost) / max(self.run_best_cost, 1.0)
                
    #             # При T=0.4..0.6 робимо лише 1 прохід. Інакше повноцінне затягування на дно.
    #             if T >= 0.4: deep_passes = 1
    #             elif gap < -0.0001: deep_passes = 10 if self.ctx.num_pipes >= 200 else 6 
    #             elif gap <= 0.002: deep_passes = 4 
    #             elif gap <= 0.02: deep_passes = 3
    #             elif gap <= 0.05: deep_passes = 2
    #             else: deep_passes = 1
                
    #             consensus_locked = set()
    #             if len(self.global_archive) >= 3 and self.is_late_game and T < 0.2 and not getattr(self, '_force_flush_next', False):
    #                 arch_sols = [x[1] for x in self.global_archive]
    #                 raw_consensus = set()
    #                 for i in range(self.ctx.num_pipes):
    #                     if all(sol[i] == arch_sols[0][i] for sol in arch_sols):
    #                         raw_consensus.add(i)
                    
    #                 freeze_pct = 0.25 if self.progress_ratio > 0.6 else 0.10
    #                 max_frozen = max(5, int(self.ctx.num_pipes * freeze_pct))
    #                 if self.progress_ratio > 0.88: max_frozen = 0
                    
    #                 if len(raw_consensus) > max_frozen and max_frozen > 0:
    #                     interesting = [i for i in raw_consensus if 0 < arch_sols[0][i] < self.ctx.max_d_idx]
    #                     consensus_locked = set(list(interesting)[:max_frozen])
    #                 elif max_frozen > 0:
    #                     consensus_locked = raw_consensus
                            
    #             safe_consensus = consensus_locked - (locked if locked else set())
    #             final_locked = locked_for_squeeze.union(safe_consensus)
                
    #             self.ctx.log(f"        [HYPERBAND] Gap {gap:.1%}. Deep Squeeze ({deep_passes} passes, {len(safe_consensus)} frozen)...")
    #             final_sol = self.ls.gradient_squeeze(quick_sol, locked_pipes=final_locked, max_passes=deep_passes, dyn_bonus=self.base_dyn_bonus)
    #         else:
    #             final_sol = quick_sol
        
    #     c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
    #     if feas and c > self.run_best_cost * 1.5:
    #         self.ctx.log(f"     -> Hard-Rejected (Cost Explosion): {c/1e6:.4f}M$")
    #         feas = False
        
    #     if feas and p >= self.ctx.simulator.config.h_min:
    #         p_surplus = p - self.ctx.simulator.config.h_min
    #         eff_bonus = self.base_dyn_bonus * 0.2 if p_surplus > 10.0 else self.base_dyn_bonus
            
    #         if p_surplus < 2.0:
    #             pre_sq_sol = final_sol 
    #             final_sol = self.ls.gradient_squeeze(final_sol, locked_pipes=set(), max_passes=3, quick_mode=True, dyn_bonus=eff_bonus)
    #             c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                
    #             if not feas or p < self.ctx.simulator.config.h_min:
    #                 self.ctx.log(f"     -> Post-squeeze became infeasible. Reverting to pre-squeeze state.")
    #                 final_sol = pre_sq_sol
    #                 c, p, feas, _ = self.ctx.get_cached_stats(final_sol)
                    
    #             p_surplus = p - self.ctx.simulator.config.h_min if feas else p_surplus
    #             eff_bonus = self.base_dyn_bonus * 0.3 if p_surplus < 1.0 else self.base_dyn_bonus
            
    #         score = c - (p_surplus * eff_bonus)
            
    #         # ==========================================
    #         # 8. ДИНАМІЧНА НАГОРОДА ТА ОНОВЛЕННЯ РЕКОРДІВ
    #         # ==========================================
    #         if c < self.run_best_cost:
    #             is_ghost = False
    #             if (self.run_best_cost - c) > (self.run_best_cost * 0.02):
    #                 is_ghost = self.ctx.is_ghost_solution(final_sol, c)
                    
    #             if is_ghost:
    #                 self.ctx.log(f"   > [SHIELD] Force illusion blocked ({c/1e6:.4f}M$)!")
    #             else:
    #                 diff = self.run_best_cost - c
    #                 self.run_best_cost, self.run_best_sol = c, final_sol
    #                 self.pool.active_pool.insert(0, (score, c, final_sol))
                    
    #                 if diff > (self.run_best_cost * 0.001): self.stagnation_counter = 0 
    #                 else: self.stagnation_counter = max(0, self.stagnation_counter - 2)
                    
    #                 improvement_pct = diff / self.run_best_cost
    #                 reward = 10.0 * improvement_pct * 100 
    #                 self.strat_wins[strategy] = self.strat_wins.get(strategy, 0) + max(1.0, reward)
                    
    #                 self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} ({self.run_best_cost/1e6:.4f}M$)")
    #                 self._update_global_best(shared_progress)

    #         elif strategy == "BASIN_ESCAPE":
    #             self.pool.active_pool.clear() 
    #             self.pool.active_pool.append((score, c, final_sol))
    #             self.stagnation_counter = 0 
    #             self.corridor_pool_streak = 0
    #             self._force_flush_next = True 
                
    #             self.run_best_cost = c
    #             self.run_best_sol = list(final_sol)
    #             self.ipc_immunity = 50
                
    #             if c < self.global_best_cost: self._update_global_best(shared_progress)
    #             self.ctx.log(f"      -> POOL FLUSHED. Adopted Major Escape: {c/1e6:.4f}M$")
                
    #         elif c < water_level and not self.pool.is_basin_tabu(final_sol):
    #             hamming_dist = self.pool.hamming_distance(final_sol, self.run_best_sol)
    #             diversity_ratio = hamming_dist / self.ctx.num_pipes
                
    #             if T >= 0.5 and diversity_ratio > 0.05:
    #                 reward = 2.0 * diversity_ratio * T
    #                 self.strat_wins[strategy] = self.strat_wins.get(strategy, 0) + reward

    #             pool_gap = (c - self.run_best_cost) / max(self.run_best_cost, 1)
    #             max_pool_gap = 0.06 if self.is_late_game else 0.12
                
    #             if self.ctx.num_pipes >= 200 and pool_gap > max_pool_gap:
    #                 self.ctx.log(f"     -> Pool Filtered (gap {pool_gap:.1%}): {c/1e6:.4f}M$")
    #             else:
    #                 self.pool.active_pool.append((score * 1.05, c, final_sol))
    #                 if strategy in ["SMART_PERTURB", "RUIN_RECREATE", "ZERO_SUM", "LOOP_BALANCE"]:
    #                     self.corridor_pool_streak = 0
    #                 self.ctx.log(f"      -> Added to Pool (Water Level Accept): {c/1e6:.4f}M$")
                    
    #         elif T >= 0.95 and not self.pool.is_basin_tabu(final_sol):
    #             self.pool.active_pool.append((score * 1.20, c, final_sol))
    #             self.stagnation_counter = int(self.stag_limit * 0.5) 
    #             self.ctx.log(f"     -> 🚀 HAIL MARY ACCEPT (Forced Escape): {c/1e6:.4f}M$")
    #         else:
    #             self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {c/1e6:.4f}M$")
            
    #         if path_sig: self.pool.kick_tabu_set.add(path_sig)
    #     else:
    #          self.ctx.log(f"     -> Injection/Squeeze Failed: Infeasible/Exploded")

    #     # 🔴 ЗОЛОТЕ ПРАВИЛО: "Забування" минулих нагород (Decay). 
    #     # Це змушує MAB адаптуватися до ПОТОЧНОЇ глибини ями!
    #     for k in list(self.strat_wins.keys()):
    #         self.strat_wins[k] *= 0.95
    #         self.strat_tries[k] = max(1.0, self.strat_tries[k] * 0.95)

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
    
    def _spatial_crossover(self, base_sol, donor_sol, T):
        """
        Епігенетичний просторовий кросовер. 
        """
        import random
        import networkx as nx
        
        if not hasattr(self.ctx, 'base_G_flow'):
            return [donor_sol[i] if random.random() < 0.5 else base_sol[i] for i in range(self.ctx.num_pipes)]

        G = self.ctx.base_G_flow
        radius = 2 + int(6.0 * T) 
        
        epicenter = random.choice(list(G.nodes()))
        local_nodes = set(nx.single_source_shortest_path_length(G, epicenter, cutoff=radius).keys())
        
        local_pipes = set()
        
        # 🔴 РОЗУМНИЙ ОБХІД РЕБЕР (БЕЗПЕЧНИЙ ДЛЯ ВСІХ ТИПІВ ГРАФІВ)
        if G.is_multigraph():
            for u, v, k in G.edges(keys=True):
                if (u in local_nodes or v in local_nodes) and isinstance(k, int):
                    local_pipes.add(k)
        else:
            for u, v, data in G.edges(data=True):
                if u in local_nodes or v in local_nodes:
                    # Шукаємо індекс труби серед атрибутів ребра
                    for val in data.values():
                        if isinstance(val, int) and 0 <= val < self.ctx.num_pipes:
                            local_pipes.add(val)
                            break
                            
        # Fallback (якщо граф побудований без збереження індексів труб)
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
        max_d = max(2, self.ctx.num_pipes // 7) 
        min_dist = max(1, int(max_d * (1.0 - (self.progress_ratio ** 2))))
        
        for rank, (score, cost, sol) in enumerate(next_gen):
            if self.pool.is_tabu(sol, cost): continue
            if len(unique_next_pool) < self.BEAM_WIDTH:
                if rank == 0: 
                    refined_sol = self.ls.gradient_squeeze(sol, max_passes=8, dyn_bonus=self.base_dyn_bonus, min_rel_improvement=0.0005)
                elif rank < 3: 
                    refined_sol = self.ls.gradient_squeeze(sol, max_passes=4, quick_mode=not self.is_late_game, dyn_bonus=self.base_dyn_bonus, min_rel_improvement=0.001)
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
                            eff_bonus = self.base_dyn_bonus * 0.2 if real_p_surplus > 10.0 else self.base_dyn_bonus
                            score = real_cost - (real_p_surplus * eff_bonus)
                        else:
                            score = real_cost + (abs(real_p_surplus) * self.base_dyn_bonus * 50.0)
                        
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
        
        forced_rescue, _, _ = self.kicker.smart_perturbation_kick(self.run_best_sol, T=1.0)
        if forced_rescue is not None:
            c, p, feas, _ = self.ctx.get_cached_stats(forced_rescue)
            if feas and p >= self.ctx.simulator.config.h_min:
                self.pool.active_pool.append((c - 1e6, c, forced_rescue))
                self.stagnation_counter = 0 
        
        if not self.pool.active_pool:
            healed_seed, _, _ = self.kicker.smart_perturbation_kick(self.run_best_sol, T=0.6)
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
        
        self.BASE_SIM_BUDGET = {"SMALL": 1000000, "MEDIUM": 3000000, "LARGE": 1500000, "XLARGE": 30000000}[self.network_class]
        self.EPOCHS = {"SMALL": 4, "MEDIUM": 4, "LARGE": 4, "XLARGE": 5}[self.network_class]
            
        if max_sims is not None:
            self.max_sims = max_sims
        else:
            self.max_sims = self.BASE_SIM_BUDGET
            
        if time_limit_sec is not None:
            self.time_limit_sec = time_limit_sec
        else:
            cluster_speed = max(self.ctx.sim_speed, 1.0) * self.n_workers
            self.time_limit_sec = (self.max_sims / cluster_speed) * 5
            
    def _build_diverse_archive(self, pool_results, target_size=6, elite_count=2):
        if not pool_results: 
            return []
            
        num_pipes = len(pool_results[0][1])
        
        unique_results = {}
        for cost, sol in pool_results:
            sig = tuple(sol)
            if sig not in unique_results or cost < unique_results[sig]:
                unique_results[sig] = cost
                
        sorted_results = sorted([(c, list(s)) for s, c in unique_results.items()], key=lambda x: x[0])
        
        archive = sorted_results[:elite_count]
        
        if num_pipes < 100:
            min_diff_pipes = max(2, int(num_pipes * 0.08))
        else:
            min_diff_pipes = max(15, min(45, int(num_pipes * 0.08)))
        
        for cost, sol in sorted_results[elite_count:]:
            if len(archive) >= target_size: 
                break
                
            min_hamming = min(sum(1 for a, b in zip(sol, arch_sol[1]) if a != b) for arch_sol in archive)
            
            if min_hamming >= min_diff_pipes:
                archive.append((cost, sol))
                
        return archive

    def solve_fast(self, ui_callback=None):
        """Швидкий аналітичний розрахунок з поліруванням ТОП-5 кандидатів."""
        print("\n[AnalyticalSolver] ⚡ Initiating FAST Analytical Search (Velocity Sweep)...\n")
        start_time = time.time()
        
        print("   > Sweeping ideal velocities (0.5 to 2.0 m/s)...")
        seeds = self.seeder.make_diverse_seeds()
        
        print("   > Sweeping Backbone (Trunk-Branch) configurations...")
        for v in [0.8, 1.0, 1.2, 1.5, 1.8]:
            bb_seed = self.seeder.make_backbone_seed(target_v=v)
            if bb_seed:
                seeds.append(bb_seed)
                
        # 1. Збираємо всі ВАЛІДНІ та УНІКАЛЬНІ зерна
        valid_seeds = []
        seen_sigs = set()
        
        for sol in seeds:
            c, p, feas, _ = self.ctx.get_cached_stats(sol)
            if feas and p >= self.ctx.simulator.config.h_min:
                sig = tuple(sol)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    valid_seeds.append((c, list(sol)))
                    
        global_best_cost = float('inf')
        global_best_sol = None
        
        # 2. ПОЛІРУЄМО НЕ ОДНОГО, А ТОП-5 ЛІДЕРІВ
        if valid_seeds:
            # Сортуємо від найдешевшого до найдорожчого
            valid_seeds.sort(key=lambda x: x[0])
            top_n = min(5, len(valid_seeds))
            
            print(f"   > Sweep finished. Found {len(valid_seeds)} valid seeds. Deep Polishing TOP-{top_n}...")
            
            for rank in range(top_n):
                raw_cost, raw_sol = valid_seeds[rank]
                
                # Глибоке полірування (False вимикає quick_mode, змушуючи перевіряти ВСЕ)
                polished = self.ls.gradient_squeeze(raw_sol, max_passes=None, quick_mode=False, dyn_bonus=raw_cost * 0.001)
                p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
                
                if p_p >= self.ctx.simulator.config.h_min and p_cost < global_best_cost:
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

    def solve_standalone(self, max_sims=None, time_limit_sec=None, ui_callback=None):
        print("\n[AnalyticalSolver] ⚡ Initiating Continuous Island Model Search...\n")
        start_time = time.time()
        global_best_cost = float('inf')
        global_best_sol = None
        global_archive = []
        
        self.history = []

        # 🔴 1. БЕЗПЕРЕРВНИЙ ЗАПУСК (Одна велика епоха)
        epochs = 1 
        time_per_epoch = self.time_limit_sec
        
        if self.max_sims == float('inf'):
            worker_epoch_sims = float('inf')
        else:
            worker_epoch_sims = int(self.max_sims // self.n_workers)
        
        quota_str = "∞" if worker_epoch_sims == float('inf') else f"{worker_epoch_sims:,}"
        print(f"  [Quota] Allocated {quota_str} sims per worker.\n")

        manager = multiprocessing.Manager() if self.mp_pool else None
        shared_progress = manager.dict() if manager else None
        if shared_progress:
            for i in range(self.n_workers): shared_progress[i] = 0
            shared_progress['global_archive'] = [] # Сховище для динамічного архіву

        cumulative_epoch_sims = 0 
        global_failed_basins = set()

        try:
            for epoch in range(epochs):
                mode_str = "PARALLEL" if self.mp_pool else "SEQUENTIAL"
                print("="*46)
                print(f" [CONTINUOUS RUN] {mode_str} Workers: {self.n_workers} | Time Limit: {time_per_epoch/60:.1f} min")
                print("="*46)
                
                seed_modifier = random.randint(1, 10000)
                tasks = []
                for i in range(self.n_workers):
                    tasks.append((
                        self.ctx.diameters, self.ctx.v_opt, time_per_epoch, 
                        global_best_cost, global_archive, 
                        seed_modifier + i, i, shared_progress, self.log_dir, epoch,
                        global_failed_basins, worker_epoch_sims, self.n_workers
                    ))

                epoch_results = []
                
                if self.mp_pool:
                    async_results = []
                    for t in tasks:
                        res = self.mp_pool.apply_async(self.worker_task, (t,))
                        async_results.append((t[6], res))

                    last_print_time = 0
                    print_interval = {"SMALL": 1.0, "MEDIUM": 15.0, "LARGE": 30.0, "XLARGE": 60.0}[self.network_class]
                    
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
                            
                            current_total = self.ctx.sim_count + cumulative_epoch_sims + total_sims
                            
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

        except KeyboardInterrupt:
            print("\n\n[AnalyticalSolver] 🛑 Отримано сигнал переривання (Ctrl+C)!")
            print("[AnalyticalSolver] М'яка зупинка. Перехід до генерації звітів...")

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
            if p_p >= self.ctx.simulator.config.h_min and p_cost < global_best_cost:
                global_best_cost = p_cost
                global_best_sol = polished
                print(f"   > [POLISH] Improved! Final: {global_best_cost/1e6:.4f}M$")
                
            current_sims = self.ctx.sim_count + cumulative_epoch_sims
            self.history.append((current_sims, global_best_cost))
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

    # def solve_standalone(self, max_sims=None, time_limit_sec=None, ui_callback=None):
    #     print("\n[AnalyticalSolver] ⚡ Initiating Island Model Search...\n")
    #     start_time = time.time()
    #     global_best_cost = float('inf')
    #     global_best_sol = None
    #     global_archive = []
        
    #     self.history = []

    #     epochs = {"SMALL": 4, "MEDIUM": 4, "LARGE": 8, "XLARGE": 8}[self.network_class]
    #     time_per_epoch = self.time_limit_sec / epochs
        
    #     if self.max_sims == float('inf'):
    #         worker_epoch_sims = float('inf')
    #     else:
    #         worker_epoch_sims = int(self.max_sims // (self.n_workers * epochs))
        
    #     quota_str = "∞" if worker_epoch_sims == float('inf') else f"{worker_epoch_sims:,}"
    #     print(f"  [Quota] Allocated {quota_str} sims per worker/epoch.\n")

    #     manager = multiprocessing.Manager() if self.mp_pool else None
    #     shared_progress = manager.dict() if manager else None
    #     if shared_progress:
    #         for i in range(self.n_workers): shared_progress[i] = 0

    #     cumulative_epoch_sims = 0 
    #     global_failed_basins = set()

    #     try:
    #         for epoch in range(epochs):
    #             mode_str = "PARALLEL" if self.mp_pool else "SEQUENTIAL"
    #             print("="*46)
    #             print(f" [EPOCH {epoch+1}/{epochs}] {mode_str} Workers: {self.n_workers} | Time Limit: {time_per_epoch/60:.1f} min")
    #             print("="*46)
                
    #             seed_modifier = random.randint(1, 10000)
    #             tasks = []
    #             for i in range(self.n_workers):
    #                 tasks.append((
    #                     self.ctx.diameters, self.ctx.v_opt, time_per_epoch, 
    #                     global_best_cost, global_archive, 
    #                     seed_modifier + i, i, shared_progress, self.log_dir, epoch,
    #                     global_failed_basins, worker_epoch_sims, self.n_workers
    #                 ))

    #             epoch_results = []
                
    #             if self.mp_pool:
    #                 async_results = []
    #                 for t in tasks:
    #                     res = self.mp_pool.apply_async(self.worker_task, (t,))
    #                     async_results.append((t[6], res))

    #                 epoch_start_time = time.time()
    #                 last_print_time = 0
    #                 print_interval = {"SMALL": 1.0, "MEDIUM": 15.0, "LARGE": 30.0, "XLARGE": 60.0}[self.network_class]
                    
    #                 while True:
    #                     all_done = all(res.ready() for _, res in async_results)
    #                     if all_done: break
                            
    #                     curr_time = time.time()
    #                     if curr_time - last_print_time >= print_interval:
    #                         last_print_time = curr_time
    #                         elapsed_total = curr_time - start_time 
    #                         m, s = divmod(int(elapsed_total), 60)
                            
    #                         status_parts = []
    #                         total_sims = 0
    #                         live_best = global_best_cost
                            
    #                         for wid in range(self.n_workers):
    #                             prog = shared_progress.get(wid, 0)
    #                             if isinstance(prog, dict):
    #                                 sims = prog.get('sims', 0)
    #                                 w_best = prog.get('best_cost', float('inf'))
    #                                 if w_best < live_best: live_best = w_best
    #                             else:
    #                                 sims = prog
    #                             total_sims += sims
    #                             if sims > 0: status_parts.append(f"W{wid+1}:{sims//1000}k")
    #                             else: status_parts.append(f"W{wid+1}:--")
                                    
    #                         status_str = " ".join(status_parts)
    #                         best_str = f"{live_best/1e6:.4f}M$" if live_best != float('inf') else "---"
    #                         print(f"   > [Live {m:02d}:{s:02d}] Best: {best_str} | Sims: {total_sims/1000:.1f}k | {status_str}")
                            
    #                         current_total = self.ctx.sim_count + cumulative_epoch_sims + total_sims
                            
    #                         if live_best != float('inf'):
    #                             self.history.append((current_total, live_best))
                                
    #                         if ui_callback is not None:
    #                             ui_callback(current_total, live_best)
                        
    #                     time.sleep(1.0)

    #                 for wid, res in async_results:
    #                     try:
    #                         c, sol, _, sims_done, worker_basins = res.get()
    #                         if sol is not None: epoch_results.append((c, sol))
    #                         global_failed_basins.update(worker_basins)
    #                     except Exception as e:
    #                         print(f"     [Error] Worker {wid+1} crashed: {e}")

    #                 if shared_progress is not None:
    #                     for wid in range(self.n_workers):
    #                         prog = shared_progress.get(wid, {})
    #                         if isinstance(prog, dict):
    #                             cumulative_epoch_sims += prog.get('sims', 0)

    #             else:
    #                 for t in tasks:
    #                     try:
    #                         c, sol, _, sims_done, worker_basins = self.worker_task(t)
    #                         if sol is not None: epoch_results.append((c, sol))
    #                         global_failed_basins.update(worker_basins)
    #                         cumulative_epoch_sims += sims_done 
    #                         print(f"   > Worker {t[6]+1} Finished. Best: {c/1e6:.4f}M$")
    #                     except Exception as e:
    #                         print(f"     [Error] Sequential Worker {t[6]+1} crashed: {e}")

    #             if epoch_results:
    #                 epoch_results_sorted = sorted(epoch_results, key=lambda x: x[0])
    #                 best_epoch_c, best_epoch_sol = epoch_results_sorted[0]
                    
    #                 if best_epoch_c < global_best_cost:
    #                     global_best_cost = best_epoch_c
    #                     global_best_sol = best_epoch_sol
    #                     print(f"\n 🏆 [EPOCH {epoch+1}] NEW GLOBAL BEST: {global_best_cost/1e6:.4f}M$ 🏆\n")
                        
    #                     current_sims = self.ctx.sim_count + cumulative_epoch_sims
    #                     self.history.append((current_sims, global_best_cost))
                        
    #                 if shared_progress is not None and 'global_best' in shared_progress:
    #                     vault_cost, vault_sol = shared_progress['global_best']
    #                     if vault_cost < global_best_cost:
    #                         global_best_cost = vault_cost
    #                         global_best_sol = list(vault_sol)
    #                         print(f"\n 🛡️ [VAULT RECOVERY] Restored historical global best: {global_best_cost/1e6:.4f}M$ 🛡️\n")
    #                         epoch_results.append((vault_cost, vault_sol))
                            
    #                         current_sims = self.ctx.sim_count + cumulative_epoch_sims
    #                         self.history.append((current_sims, global_best_cost))

    #                 global_archive = self._build_diverse_archive(epoch_results, target_size=6, elite_count=2)
                    
    #             if epoch < epochs - 1 and len(global_archive) < 6:
    #                 missing_slots = 6 - len(global_archive)
    #                 print(f"   [DIVERSITY CHECK] Found {len(global_archive)} unique structural basins. Injecting {missing_slots} cold seeds.")
    #                 cold_seeds = self.seeder.make_diverse_seeds()
                    
    #                 for cs in cold_seeds:
    #                     c, _, feas, _ = self.ctx.get_cached_stats(cs)
    #                     if feas: 
    #                         global_archive.append((c, cs))
    #                         if len(global_archive) >= 6: break
                    
    #                 global_archive = sorted(global_archive, key=lambda x: x[0])[:6]

    #     except KeyboardInterrupt:
    #         print("\n\n[AnalyticalSolver] 🛑 Отримано сигнал переривання (Ctrl+C)!")
    #         print("[AnalyticalSolver] М'яка зупинка епох. Перехід до генерації звітів...")

    #     if shared_progress is not None:
    #         gb = shared_progress.get('global_best')
    #         if gb and gb[0] < global_best_cost:
    #             global_best_cost = gb[0]
    #             global_best_sol = list(gb[1])
                
    #         for wid in range(self.n_workers):
    #             w_sol = shared_progress.get(f'best_sol_{wid}')
    #             if w_sol and w_sol[0] < global_best_cost:
    #                 global_best_cost = w_sol[0]
    #                 global_best_sol = list(w_sol[1])

    #     print("\n[FINAL POLISH] Polishing global best solution...")
    #     if global_best_sol:
    #         polished = self.ls.gradient_squeeze(global_best_sol, max_passes=None, quick_mode=False, dyn_bonus=global_best_cost * 0.001)
    #         p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
    #         if p_p >= self.ctx.simulator.config.h_min and p_cost < global_best_cost:
    #             global_best_cost = p_cost
    #             global_best_sol = polished
    #             print(f"   > [POLISH] Improved! Final: {global_best_cost/1e6:.4f}M$")
                
    #             current_sims = self.ctx.sim_count + cumulative_epoch_sims
    #             self.history.append((current_sims, global_best_cost))
    #     else:
    #         print("\n[WARNING] No valid solution found. Returning safe default.")
    #         global_best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
    #         try: global_best_cost, _, _, _ = self.ctx.get_cached_stats(global_best_sol)
    #         except: global_best_cost = float('inf')
        
    #     total_time = time.time() - start_time
    #     total_cluster_sims = self.ctx.sim_count + cumulative_epoch_sims

    #     if global_best_cost != float('inf'):
    #         print(f"\n[AnalyticalSolver] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Total Time: {total_time/60:.1f}m | Total Sims: {total_cluster_sims:,})")
    #     else:
    #         print(f"\n[AnalyticalSolver] EXECUTION ABORTED. No valid solutions.")
        
    #     real_diams = [self.ctx.diameters[i] for i in global_best_sol] if global_best_sol else []
    #     return real_diams