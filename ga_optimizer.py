import random
import time
import statistics
import networkx as nx
import numpy as np
from deap import base, creator, tools
from ga_config import CONFIG
from ga_utils import format_time

class GeneticOptimizer:
    def __init__(self, simulator, pop_size=None, n_gens=None, pool=None, fixed_mode=False):
        self.sim = simulator
        self.n_vars = simulator.n_variables
        self.n_opts = simulator.n_options
        self.memo = {} 
        self.start_time = time.time()
        self.pool = pool 
        self.fixed_mode = fixed_mode

        self.total_sims = 0
        
        self._calibrate_penalties()
        
        print("\n--- [System] Auto-Configuration ---")
        if not pop_size or pop_size < 10:
            self.pop_size = max(200, min(800, int(self.n_vars * 5.0)))
            print(f"[Auto-Config] Population: {self.pop_size}")
        else:
            self.pop_size = pop_size
            print(f"[Config] Population: {self.pop_size}")

        if not n_gens or n_gens < 10:
            base_gens = 1500
            scale_factor = self.n_vars * 10
            self.n_gens = base_gens + scale_factor
            print(f"[Auto-Config] Max Generations: {self.n_gens}")
        else:
            self.n_gens = n_gens
            print(f"[Config] Max Generations: {self.n_gens}")

        self.plateau_patience = 50 
        print(f"[Config] Mode: Deep Dive with post shock")
        print(f"[Config] Penalties: Linear (Factor={self.HEAVY_PF:.0f})")
        print("-----------------------------------\n")

        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        if hasattr(creator, "Individual"): del creator.Individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.15)
        self.toolbox.register("attr_int", random.randint, 0, self.n_opts-1)

    def _get_stats(self, ind):
        self.total_sims += 1
        return self.sim.get_stats(ind)

    def _get_heuristics(self, ind):
        self.total_sims += 1
        return self.sim.get_heuristics(ind)

    def _calibrate_penalties(self):
        costs = list(CONFIG['costs'].values())
        avg_cost = statistics.mean(costs) if costs else 100.0
        self.HEAVY_PF = avg_cost * 5000.0  
        self.DEATH_LIMIT = 1e16 

    def _get_diversity(self, pop):
        if not pop: return 0.0
        return len(set(tuple(ind) for ind in pop)) / len(pop)

    def _inject_diversity(self, pop, ratio=0.2):
        n_replace = int(len(pop) * ratio)
        pop.sort(key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('inf'))
        for i in range(len(pop)-n_replace, len(pop)):
            ind = [random.randint(0, self.n_opts-1) for _ in range(self.n_vars)]
            pop[i] = creator.Individual(ind)
            if hasattr(pop[i], "fitness"): del pop[i].fitness.values
        return pop

    def _expand_population(self, pop):
        add_count = int(len(pop) * 0.3)
        print(f"\n    [EXPANSION] 🚀 Boosting population: {len(pop)} -> {len(pop) + add_count}")
        for _ in range(add_count):
            pop.append(creator.Individual([random.randint(1, self.n_opts-1) for _ in range(self.n_vars)]))
        return pop

    def _hamming_distance(self, ind1, ind2):
        return sum(1 for i in range(len(ind1)) if ind1[i] != ind2[i])

    def _apply_fitness_sharing(self, pop, penalty_factor=0.1, banned_cost=None):
        counts = {}
        for ind in pop:
            if not ind.fitness.valid: continue
            k = tuple(ind)
            counts[k] = counts.get(k, 0) + 1
        for ind in pop:
            if not ind.fitness.valid: continue
            if banned_cost:
                if abs(ind.fitness.values[0] - banned_cost) < (banned_cost * 0.002):
                     ind.fitness.values = (self.DEATH_LIMIT * 0.9,) 
                     continue
            if counts[tuple(ind)] > 1 and ind.fitness.values[0] < self.DEATH_LIMIT:
                ind.fitness.values = (ind.fitness.values[0] * (1.0 + penalty_factor * (counts[tuple(ind)] - 1)),)

    def _smart_mutation(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = max(0, min(individual[i] + random.choice([-1, 1]), self.n_opts - 1))
        return individual,

    def _path_mutation(self, individual, shrink_bias=0.5):
        if random.random() < shrink_bias:
            candidates = [i for i, v in enumerate(individual) if v > 0]
            if candidates: individual[random.choice(candidates)] -= 1
        else:
            idx = random.randint(0, self.n_vars - 1)
            if individual[idx] < self.n_opts - 1: individual[idx] += 1
        return individual,

    def _cached_eval(self, individual, gen, pf, epsilon):
        key = (tuple(individual), int(pf), round(epsilon, 3))
        if key in self.memo: return self.memo[key]
        
        self.total_sims += 1
        val = self.sim.evaluate(individual, penalty_factor=pf, epsilon=epsilon)
        
        if not isinstance(val, tuple): val = (val,)
        self.memo[key] = val
        return val
    
    def _random_warmup(self, start_eps, start_pf):
        print(f"    [Warmup] 🎲 Using PURE RANDOM strategy...")
        
        pop = []
        for _ in range(self.pop_size):
            ind = creator.Individual([random.randint(0, self.n_opts-1) for _ in range(self.n_vars)])
            pop.append(ind)
            
        if self.pool:
            tasks = [(ind, 0, start_pf, start_eps) for ind in pop]
            self.total_sims += len(tasks)
            results = self.pool.map(self.sim.worker_eval_wrapper, tasks)
            for ind, fit in zip(pop, results): 
                if not isinstance(fit, tuple): fit = (fit,)
                ind.fitness.values = fit
        else:
            for ind in pop: 
                ind.fitness.values = self._cached_eval(ind, 0, start_pf, start_eps)
                
        valid_count = sum(1 for ind in pop if ind.fitness.values[0] < self.DEATH_LIMIT)
        print(f"    [Warmup] ✅ Generated {len(pop)} items. Valid initial solutions: {valid_count}")
        
        return pop
    
    def _guaranteed_warmup(self, start_eps, start_pf):
        print(f"    [Warmup] 🛡️ Using GUARANTEED strategy (20% Max, 30% Mid, 50% Rnd)...")
        
        pop = []
        n_max = int(self.pop_size * 0.20)
        n_mid = int(self.pop_size * 0.30)
        n_rnd = self.pop_size - n_max - n_mid
        max_idx = self.n_opts - 1
        for _ in range(n_max):
            ind = creator.Individual([max_idx] * self.n_vars)
            pop.append(ind)

        mid_idx = self.n_opts // 2
        for _ in range(n_mid):

            ind = creator.Individual([
                max(0, min(mid_idx + random.randint(-1, 1), self.n_opts - 1)) 
                for _ in range(self.n_vars)
            ])
            pop.append(ind)

        for _ in range(n_rnd):
            ind = creator.Individual([random.randint(0, self.n_opts - 1) for _ in range(self.n_vars)])
            pop.append(ind)
            
        if self.pool:
            tasks = [(ind, 0, start_pf, start_eps) for ind in pop]
            self.total_sims += len(tasks)
            results = self.pool.map(self.sim.worker_eval_wrapper, tasks)
            for ind, fit in zip(pop, results): 
                if not isinstance(fit, tuple): fit = (fit,)
                ind.fitness.values = fit
        else:
            for ind in pop: 
                ind.fitness.values = self._cached_eval(ind, 0, start_pf, start_eps)
            
        valid_count = sum(1 for ind in pop if ind.fitness.values[0] < self.DEATH_LIMIT)
        print(f"    [Warmup] ✅ Generated {len(pop)} items. Valid initial solutions: {valid_count}")
        
        return pop

    def _mixed_warmup(self, start_eps, start_pf):
        warmup_pop_size = self.pop_size * 10
        warmup_gens = 30
        print(f"    [Hyper-Warmup] 🛡️ Generating mixed initial skeletons...")
        
        pop = []
        for _ in range(int(warmup_pop_size * 0.5)):
            ind = creator.Individual([random.randint(2, self.n_opts-1) for _ in range(self.n_vars)])
            pop.append(ind)
        for _ in range(int(warmup_pop_size * 0.5)):
            ind = creator.Individual([random.randint(2, 4) for _ in range(self.n_vars)])
            pop.append(ind)
            
        if self.pool:
            tasks = [(ind, 0, start_pf, start_eps) for ind in pop]
            self.total_sims += len(tasks)
            results = self.pool.map(self.sim.worker_eval_wrapper, tasks)
            for ind, fit in zip(pop, results): 
                if not isinstance(fit, tuple): fit = (fit,)
                ind.fitness.values = fit
        else:
            for ind in pop: ind.fitness.values = self._cached_eval(ind, 0, start_pf, start_eps)
            
        for g in range(warmup_gens):
            offspring = tools.selTournament(pop, len(pop), tournsize=2)
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8: self.toolbox.mate(child1, child2); del child1.fitness.values, child2.fitness.values
            for ind in offspring:
                if random.random() < 0.3: self._path_mutation(ind, shrink_bias=0.5); del ind.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                if self.pool:
                    tasks = [(ind, 0, start_pf, start_eps) for ind in invalid_ind]
                    self.total_sims += len(tasks)
                    results = self.pool.map(self.sim.worker_eval_wrapper, tasks)
                    for ind, fit in zip(invalid_ind, results): 
                        if not isinstance(fit, tuple): fit = (fit,)
                        ind.fitness.values = fit
                else:
                    for ind in invalid_ind: ind.fitness.values = self._cached_eval(ind, 0, start_pf, start_eps)
            pop[:] = offspring
            if g % 10 == 0: 
                raw_costs = [ind.fitness.values[0] for ind in pop]
                best_raw = min(raw_costs)
                tag = "(Valid)" if best_raw < self.DEATH_LIMIT else "(Inv)"
                disp = best_raw if best_raw < self.DEATH_LIMIT else (best_raw - self.DEATH_LIMIT)
                if disp > 1e9: disp = best_raw 
                print(f"      > Warmup Gen {g}: Best {disp/1e6:.2f}M$ {tag}")

        print("    [Hyper-Warmup] 🧬 Selecting Zombies & Heroes...")
        pop.sort(key=lambda ind: ind.fitness.values[0])
        heroes = pop[:int(self.pop_size * 0.5)]
        pop.sort(key=lambda ind: sum(ind)) 
        zombies = pop[:int(self.pop_size * 0.5)]
        unique_pop = []
        seen = set()
        for ind in heroes + zombies:
            t = tuple(ind)
            if t not in seen: seen.add(t); unique_pop.append(ind)
        while len(unique_pop) < self.pop_size:
             ind = creator.Individual([random.randint(1, self.n_opts-1) for _ in range(self.n_vars)])
             ind.fitness.values = (self.DEATH_LIMIT + 999,); unique_pop.append(ind)
        print(f"    [Hyper-Warmup] ✅ Ready to dive with {len(unique_pop)} diverse agents.")
        return unique_pop

    def _smart_repair(self, individual, epsilon, max_steps=20):
        current = list(individual)
        eff_limit = CONFIG['h_min'] - epsilon
        
        _, init_p, _, _ = self._get_stats(current)
        if init_p >= eff_limit: return current, "NO-OP"

        for boost_level in range(15):
            for _ in range(5):
                _, min_p, _, crit_node = self._get_stats(current)
                if min_p >= eff_limit: 
                    tag = "PATH_FIX" if boost_level == 0 else f"BST+{boost_level}"
                    return current, tag

                try: path = nx.shortest_path(self.sim.graph, self.sim.sources[0], crit_node)
                except: 
                    r = random.randint(0, self.n_vars-1)
                    current[r] = min(current[r] + 1, self.n_opts - 1)
                    continue

                path_indices = []
                for u, v in zip(path[:-1], path[1:]):
                    edge = self.sim.graph.get_edge_data(u, v)
                    for link in edge:
                        if link in self.sim.component_names:
                            path_indices.append(self.sim.component_names.index(link))
                            break
                if not path_indices: break 

                unit_losses = self._get_heuristics(current)
                path_indices.sort(key=lambda idx: unit_losses[idx], reverse=True)
                changed = False
                for idx in path_indices[:3]:
                    if current[idx] < self.n_opts - 1: current[idx] += 1; changed = True
                if not changed: break 

            any_boosted = False
            for i in range(self.n_vars):
                if current[i] < self.n_opts - 1: current[i] += 1; any_boosted = True
            if not any_boosted: return current, "MAXED"

        return current, "ITER_BST"

    def run_local_search(self, individual, limit_pipes=None, epsilon=0.0):
        current = list(individual)
        eff_limit = CONFIG['h_min'] - epsilon
        unit_losses = self._get_heuristics(current)
        for _ in range(5):
            cost, min_p, _, _ = self._get_stats(current)
            if min_p < eff_limit: return current 
            diams_raw = CONFIG["diameters_raw"]
            costs = CONFIG["costs"]
            scores = []
            for i in range(self.n_vars):
                if current[i] > 0:
                    d = diams_raw[current[i]]
                    loss = unit_losses[i] + 1e-9
                    scores.append((i, costs[d] / loss))
            scores.sort(key=lambda x: x[1], reverse=True)
            improved = False
            for i, _ in scores[:15]:
                cand = list(current)
                cand[i] -= 1
                c_new, p_new, _, _ = self._get_stats(cand)
                if p_new >= eff_limit and c_new < cost:
                    current = cand
                    improved = True
                    unit_losses = self._get_heuristics(current)
                    break 
            if not improved: break
        return current

    def _iterative_squeeze(self, individual, epsilon=0.0, silent=False):
        if not silent: print(f"    [Polishing] Starting Iterative Squeeze (Eps={epsilon:.2f})...")
        current = list(individual)
        eff_limit = CONFIG['h_min'] - epsilon
        while True:
            indices = sorted(range(self.n_vars), key=lambda i: current[i], reverse=True)
            improved = False
            for i in indices:
                if current[i] > 0:
                    orig = current[i]
                    current[i] -= 1
                    _, p, _, _ = self._get_stats(current)
                    if p >= eff_limit:
                        improved = True
                    else: current[i] = orig
            if not improved: break
        return current

    def _invalidate_fitness(self, pop, hof):
        for ind in pop: 
            if hasattr(ind, "fitness"): del ind.fitness.values
        for ind in hof: 
            if hasattr(ind, "fitness"): del ind.fitness.values

    def run(self, run_id, h_min, 
            init_mode='hyper',
            use_epsilon=True, 
            use_expansion=True, 
            use_shocks=True, 
            use_graph_heuristics=True):
        
        random.seed(time.time() + run_id)
        np.random.seed(int(time.time() + run_id))
        self.start_time = time.time()
        print(f"\n>>> Starting Run #{run_id + 1}...")
        print(f"    [Config] Init: {init_mode.upper()} | Eps: {use_epsilon} | Shocks: {use_shocks} | Expand: {use_expansion} | Graphs: {use_graph_heuristics}")

        CONFIG['h_min'] = h_min 
        current_pf = self.HEAVY_PF
        
        if use_epsilon:
            start_eps = h_min * 0.50 
            steps = [h_min * x for x in [0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.0]]
        else:
            start_eps = 0.0
            steps = [0.0]
            
        if init_mode == 'static':
            pop = self._guaranteed_warmup(start_eps, current_pf)
        elif init_mode == 'random':
            pop = self._random_warmup(start_eps, current_pf)
        else:
            pop = self._mixed_warmup(start_eps, current_pf)
            
        hof = tools.HallOfFame(10)
        
        step_idx = 0
        current_epsilon = start_eps
        
        hof.update(pop)

        history = []
        last_best_fitness = hof[0].fitness.values[0]
        stagnation_counter = 0
        gen = 0
        shock_active = False
        cooldown_active = False 
        shock_end_gen = 0
        shock_levels = [h_min * x for x in [0.20, 0.30, 0.40]]
        
        cost_before_shock = float('inf')
        cooldown_steps = []
        cooldown_idx = 0
        expansions_used = 0 
        shock_triggered_count = 0
        banned_legacy_cost = None
        current_mutation_rate = 0.05
        
        consecutive_fails = 0
        MAX_TOTAL_SHOCKS = 10 if use_shocks else 0
        MAX_FAILS = 5 
        record_at_shock_start = float('inf')
        
        epsilon_registry = {} 
        global_valid_cost = float('inf')
        global_valid_ind = None

        self.toolbox.register("mutate", self._smart_mutation, indpb=current_mutation_rate)

        while (gen < self.n_gens) or (shock_active or cooldown_active) or (current_epsilon == 0.0 and consecutive_fails < MAX_FAILS):
            if gen > self.n_gens * 5: break
            if gen >= self.n_gens and not shock_active and not cooldown_active:
                print(f"    [Auto-Stop] 🛑 Reached strict generation limit ({self.n_gens}).")
                break

            elapsed = time.time() - self.start_time
            if gen > 20:
                eta = (elapsed / gen) * (self.n_gens - gen) if gen < self.n_gens else 0
                eta_str = format_time(eta)
            else: eta_str = "..."

            # --- 1. LOGIC PHASE ---
            diversity = self._get_diversity(pop)
            
            if diversity < 0.2: current_mutation_rate = 0.20
            elif diversity < 0.4: current_mutation_rate = 0.10
            else: current_mutation_rate = 0.05
            self.toolbox.register("mutate", self._smart_mutation, indpb=current_mutation_rate)

            current_best_fitness = hof[0].fitness.values[0]
            
            if abs(current_best_fitness - last_best_fitness) < 10.0: stagnation_counter += 1
            else: stagnation_counter = 0; last_best_fitness = current_best_fitness

            curr_c, leader_p, _, _ = self._get_stats(hof[0])
            eff_limit = h_min - current_epsilon
            
            if leader_p >= eff_limit:
                eps_key = round(current_epsilon, 2)
                if eps_key not in epsilon_registry or curr_c < epsilon_registry[eps_key][1]:
                    epsilon_registry[eps_key] = (creator.Individual(hof[0]), curr_c)
            
            if leader_p >= h_min:
                if curr_c < global_valid_cost:
                    print(f"    [Record] 🏆 New Global Best: {curr_c/1e6:.4f}M$ (Valid)")
                    global_valid_cost = curr_c
                    global_valid_ind = creator.Individual(hof[0])
                    consecutive_fails = 0 
            
            is_valid_now = leader_p >= eff_limit

            if not is_valid_now:
                current_pf *= 1.2
                self._invalidate_fitness(pop, hof)
                status_msg = "INVALID (PF↑)"
                
                if stagnation_counter > 2:
                    if use_graph_heuristics:
                        steps_fix = 5 if current_epsilon > (h_min * 0.4) else 40
                        fixed_ind, method = self._smart_repair(hof[0], current_epsilon, max_steps=steps_fix)
                        if "BST" in method or method == "ITER_BST":
                            fixed_ind = self._iterative_squeeze(fixed_ind, epsilon=current_epsilon, silent=True)
                            method += "+SQZ"
                        pop[0] = creator.Individual(fixed_ind)
                        del pop[0].fitness.values
                        status_msg = f"{method}"
                        stagnation_counter = 0
                    else:
                        fixed_ind = list(hof[0])
                        for _ in range(5):
                            idx = random.randint(0, self.n_vars - 1)
                            if fixed_ind[idx] < self.n_opts - 1: fixed_ind[idx] += 1
                        method = "NAIVE_BUMP"
                        pop[0] = creator.Individual(fixed_ind)
                    
                    del pop[0].fitness.values
                    status_msg = f"{method}"
                    stagnation_counter = 0
            else:
                current_pf = max(self.HEAVY_PF, current_pf * 0.95)
                status_msg = "OK"

            if  use_graph_heuristics and gen % 30 == 0 and is_valid_now and not shock_active and not cooldown_active and (current_epsilon < h_min * 0.2):
                cand = list(hof[0])
                squeezed_cand = self._iterative_squeeze(cand, epsilon=current_epsilon, silent=True)
                new_ind = creator.Individual(squeezed_cand)
                fit_val = self._cached_eval(new_ind, gen, current_pf, current_epsilon)
                new_ind.fitness.values = fit_val
                
                sq_cost, sq_p, _, _ = self.sim.get_stats(new_ind)

                if sq_p >= h_min:
                    if sq_cost < global_valid_cost:
                        print(f"    [Record] 🏆 New Global Best (from Squeeze): {sq_cost/1e6:.4f}M$ (Valid)")
                        global_valid_cost = sq_cost
                        global_valid_ind = creator.Individual(new_ind)
                        consecutive_fails = 0

                if new_ind.fitness.values[0] < hof[0].fitness.values[0]:
                    print(f"    [Optimization] 📉 Squeeze: {hof[0].fitness.values[0]/1e6:.4f}M -> {new_ind.fitness.values[0]/1e6:.4f}M")
                    hof.update([new_ind]); pop[0] = new_ind

            # --- 2. TRANSITIONS ---
            if not shock_active and not cooldown_active:
                if stagnation_counter > self.plateau_patience:
                    if step_idx < len(steps):
                        next_eps = steps[step_idx]
                        print(f"    [Handover] Switching to Eps {next_eps:.2f}m...")
                        
                        next_key = round(next_eps, 2)
                        if next_key in epsilon_registry:
                            reg_ind, reg_cost = epsilon_registry[next_key]
                            print(f"      > Restored champion for Eps {next_eps} from registry ({reg_cost/1e6:.2f}M$)")
                            repaired_leader = creator.Individual(reg_ind)
                        else:
                            print(f"      > Creating repaired clone...")
                            if use_graph_heuristics:
                                rep_genes, _ = self._smart_repair(hof[0], next_eps, max_steps=50)
                                rep_genes = self._iterative_squeeze(rep_genes, epsilon=next_eps, silent=True)
                            else:
                                rep_genes = list(hof[0])
                                for _ in range(5):
                                    idx = random.randint(0, self.n_vars - 1)
                                    if rep_genes[idx] < self.n_opts - 1: rep_genes[idx] += 1
                            repaired_leader = creator.Individual(rep_genes)
                        
                        current_epsilon = next_eps
                        step_idx += 1
                        stagnation_counter = 0 
                        
                        self._invalidate_fitness(pop, hof)
                        pop[0] = repaired_leader
                        if hasattr(pop[0], 'fitness'): del pop[0].fitness.values
                        
                        hof.update([repaired_leader])
                        pop = self._inject_diversity(pop, ratio=0.25)

                    elif current_epsilon == 0.0:
                        if  use_shocks and shock_triggered_count >= MAX_TOTAL_SHOCKS:
                            print(f"    [Auto-Stop] 🛑 Reached max total shocks ({MAX_TOTAL_SHOCKS}). Stopping.")
                            break
                        elif not use_shocks:
                            print(f"    [Auto-Stop] 🛑 Stagnation reached 0.0 limit without shocks. Stopping.")
                            break
                        
                        if consecutive_fails >= MAX_FAILS:
                            print(f"    [Auto-Stop] 🛑 Too many consecutive fails ({consecutive_fails}). Stopping.")
                            break

                        if use_expansion and stagnation_counter > 50 and expansions_used < 2:
                             pop = self._expand_population(pop)
                             expansions_used += 1; stagnation_counter = 0
                             self._invalidate_fitness(pop, hof); continue

                        if use_shocks:
                            shock_active = True; shock_triggered_count += 1
                            record_at_shock_start = global_valid_cost
                            shock_val = shock_levels[(shock_triggered_count - 1) % len(shock_levels)]    
                            
                            shock_duration = 40 + int(self.n_vars * 0.25)
                            shock_duration = max(50, min(300, shock_duration))
                            shock_end_gen = gen + shock_duration
                            
                            cost_before_shock = hof[0].fitness.values[0]
                            banned_legacy_cost = cost_before_shock
                            stagnation_counter = 0
                            current_epsilon = shock_val 
                            self._invalidate_fitness(pop, hof); hof.clear() 
                            print(f"\n    [Seismic] 〰️ Deep Shock #{shock_triggered_count}! Relaxing to Eps -> {shock_val:.2f}m.")

            elif shock_active:
                current_epsilon = shock_val
                status_msg = f"SHOCK-{shock_triggered_count}"
                if gen >= shock_end_gen:
                    shock_active = False; cooldown_active = True
                    cooldown_steps = [shock_val * x for x in [0.8, 0.6, 0.4, 0.2, 0.0]]
                    cooldown_idx = 0; current_epsilon = cooldown_steps[0]
                    stagnation_counter = 0
                    print(f"\n    [Seismic] Shock ended. Anti-Atavism engaged.")
                    print(f"    [Cooldown] 🌡️ Cooling: Eps -> {current_epsilon:.2f}m")
                    self._invalidate_fitness(pop, hof); hof.clear(); hof.update(pop)

            elif cooldown_active:
                status_msg = f"COOLING (BAN {banned_legacy_cost/1e6:.4f})" if banned_legacy_cost else "COOLING"
                if stagnation_counter > 50:
                    cooldown_idx += 1
                    if cooldown_idx < len(cooldown_steps):
                        current_epsilon = cooldown_steps[cooldown_idx]
                        stagnation_counter = 0
                        print(f"    [Cooldown] 🌡️ Cooling: Eps -> {current_epsilon:.2f}m")
                        self._invalidate_fitness(pop, hof)
                        
                        next_key = round(current_epsilon, 2)
                        if next_key in epsilon_registry:
                             reg_ind, _ = epsilon_registry[next_key]
                             pop[0] = creator.Individual(reg_ind)
                        else:
                            if use_graph_heuristics:
                                rep_genes, _ = self._smart_repair(hof[0], current_epsilon, max_steps=30)
                                rep_genes = self._iterative_squeeze(rep_genes, epsilon=current_epsilon, silent=True)
                            else:
                                rep_genes = list(hof[0])
                                for _ in range(5):
                                    idx = random.randint(0, self.n_vars - 1)
                                    if rep_genes[idx] < self.n_opts - 1: rep_genes[idx] += 1
                            pop[0] = creator.Individual(rep_genes)
                        del pop[0].fitness.values
                    else:
                        cooldown_active = False; current_epsilon = 0.0
                        banned_legacy_cost = None
                        
                        current_best_fitness = hof[0].fitness.values[0]
                        
                        if global_valid_cost < record_at_shock_start:
                            consecutive_fails = 0
                            print(f"    [Cooldown] 🚀 NEW RECORD confirmed! ({global_valid_cost/1e6:.4f}M$). Strikes reset.")
                        else:
                            consecutive_fails += 1
                            print(f"    [Cooldown] 📉 No new record. Strike {consecutive_fails}/{MAX_FAILS}.")

                        print(f"    [Cooldown] ✅ Soft Landing complete (0.0m).")
                        self._invalidate_fitness(pop, hof)

            # --- 3. EVALUATION ---
            if self.pool:
                invalids = [ind for ind in pop if not ind.fitness.valid]
                invalids.append(pop[0]); 
                if len(pop)>1: invalids.append(pop[1])
                to_eval = []
                seen_ids = set()
                for ind in invalids:
                    tid = tuple(ind)
                    if tid not in seen_ids: to_eval.append(ind); seen_ids.add(tid)
                tasks = [(ind, gen, current_pf, current_epsilon) for ind in to_eval]
                self.total_sims += len(tasks)
                results = self.pool.map(self.sim.worker_eval_wrapper, tasks)
                res_map = {tuple(ind): res for ind, res in zip(to_eval, results)}
                for ind in invalids:
                    val = res_map.get(tuple(ind), (float('inf'),))
                    if not isinstance(val, tuple): val = (val,) 
                    ind.fitness.values = val
            else:
                for ind in pop:
                    if not ind.fitness.valid: ind.fitness.values = self._cached_eval(ind, gen, current_pf, current_epsilon)

            # --- 4. STATS ---
            self._apply_fitness_sharing(pop, penalty_factor=0.2, banned_cost=banned_legacy_cost)
            if banned_legacy_cost:
                safe_pop = [ind for ind in pop if abs(ind.fitness.values[0] - banned_legacy_cost) > 5000]
                if safe_pop: hof.update(safe_pop)
            else: hof.update(pop)

            curr_c, curr_min_p, _, _ = self._get_stats(hof[0])
            history.append({'gen': gen, 'cost': curr_c, 'min_pressure': curr_min_p})

            if gen % 10 == 0:
                print(f"    [Gen {gen:4d}] Cost={curr_c/1e6:.4f}M$ | P={leader_p:.2f}m | Eps={current_epsilon:.3f} | Sims={self.total_sims} | {status_msg} | Time: {format_time(elapsed)} (ETA: {eta_str})")

            # --- 5. EVOLUTION ---
            offspring = []
            for _ in range(len(pop)):
                allow_zombies = (not cooldown_active and not shock_active) and (random.random() < 0.3)
                if allow_zombies: winner = min(random.sample(pop, 2), key=lambda ind: sum(ind))
                else: winner = tools.selTournament(pop, 1, tournsize=2)[0]
                offspring.append(self.toolbox.clone(winner))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8: self.toolbox.mate(child1, child2); del child1.fitness.values, child2.fitness.values
            for ind in offspring:
                if random.random() < 0.5: self.toolbox.mutate(ind); del ind.fitness.values
            pop[:] = offspring
            gen += 1

        print("\n    [Finalizing] Validating & Polishing...")
       
        final_cost, final_p, _, _ = self._get_stats(hof[0])
        
        if global_valid_ind is not None:
            if final_p < CONFIG['h_min'] or final_cost > global_valid_cost:
                print(f"    [Memory] ⏪ Restoring GLOBAL best valid solution ({global_valid_cost/1e6:.4f}M$)")
                hof.clear(); hof.update([global_valid_ind])
        
        if use_graph_heuristics:
            semi_final = self.run_local_search(hof[0], limit_pipes=None, epsilon=0.0)
            final_solution = self._iterative_squeeze(semi_final)
        else:
            final_solution = list(hof[0])
            
        final_ind = creator.Individual(final_solution)
        cost, min_p, _, _ = self._get_stats(final_ind)
        
        total_time = time.time() - self.start_time
        print(f"    >> Done in {format_time(total_time)}. Final Result: {cost/1e6:.4f}M$ | P={min_p:.2f}m | Sims={self.total_sims} | Time: {format_time(total_time)}")
        return final_ind, cost, min_p, total_time, history