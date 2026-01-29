import random
import time
from deap import base, creator, tools
from ga_config import CONFIG, MUTATION_START, MUTATION_END, EPSILON_START, EPSILON_END
from ga_utils import format_time

class GeneticOptimizer:
    def __init__(self, simulator, pop_size, n_gens):
        self.sim = simulator
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.n_diams = len(CONFIG["diameters_raw"])
        self.n_pipes = simulator.n_pipes
        self.memo = {} 
        self.start_time = time.time()

        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        if hasattr(creator, "Individual"): del creator.Individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        # Uniform Crossover (indpb=0.20) - краще для довгих векторів
        self.toolbox.register("mate", tools.cxUniform, indpb=0.20)
        self.toolbox.register("attr_int", random.randint, 0, self.n_diams-1)

    def _smart_mutation(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                r = random.random()
                if r < 0.85: 
                    step = random.choice([-1, 1])
                    new_val = individual[i] + step
                    if new_val < 0: new_val = 0
                    elif new_val > self.n_diams - 1: new_val = self.n_diams - 1
                    individual[i] = new_val
                else:
                    individual[i] = random.randint(0, self.n_diams - 1)
        return individual,

    def _apply_hyper_mutation(self, population, strength=0.2):
        genes_to_change = int(self.n_pipes * strength)
        if genes_to_change < 1: genes_to_change = 1
        
        for ind in population:
            indices = random.sample(range(self.n_pipes), genes_to_change)
            for idx in indices:
                # 50/50: або товсті, або випадкові
                if random.random() < 0.5:
                    ind[idx] = random.randint(self.n_diams // 2, self.n_diams - 1)
                else:
                    ind[idx] = random.randint(0, self.n_diams - 1)
            del ind.fitness.values
        return population

    def _create_population(self):
        pop = []
        # Feasibility First Init
        # 1. Max (20%)
        for _ in range(int(self.pop_size * 0.20)):
            pop.append(creator.Individual([self.n_diams - 1] * self.n_pipes))
        # 2. Top-Half (30%)
        mid_idx = self.n_diams // 2
        for _ in range(int(self.pop_size * 0.30)):
             pop.append(creator.Individual([random.randint(mid_idx, self.n_diams-1) for _ in range(self.n_pipes)]))
        # 3. Median (30%)
        for _ in range(int(self.pop_size * 0.30)):
            pop.append(creator.Individual([mid_idx] * self.n_pipes))
        # 4. Random (20%)
        remaining = self.pop_size - len(pop)
        for _ in range(remaining):
            pop.append(creator.Individual([random.randint(0, self.n_diams-1) for _ in range(self.n_pipes)]))
        return pop

    def _cached_eval(self, individual, gen, cache_enabled=True):
        limit_gen = self.n_gens * 0.90
        if gen < limit_gen:
            progress = gen / limit_gen
            factor = (1.0 - progress) ** 2 
            tol = max(0.0, EPSILON_START * factor)
        else: 
            tol = 0.0

        if not cache_enabled:
            return self.sim.evaluate(individual, "epsilon", gen, tol, self.n_gens)

        ind_tuple = tuple(individual)
        if ind_tuple in self.memo: return self.memo[ind_tuple]
        
        fit = self.sim.evaluate(individual, "epsilon", gen, tol, self.n_gens)
        self.memo[ind_tuple] = fit
        return fit

    def run_local_search(self, individual, limit_pipes=None, verbose=False):
        start = time.time()
        current = list(individual)
        cost_penalty_tuple = self.sim.evaluate(current, "static", tolerance=0.0)
        current_score = cost_penalty_tuple[0]
        
        is_repair_mode = (current_score > 1e9)
        
        indices = list(range(self.n_pipes))
        if limit_pipes and limit_pipes < self.n_pipes:
            if not is_repair_mode:
                indices.sort(key=lambda i: current[i], reverse=True)
            else:
                random.shuffle(indices)
            indices = indices[:limit_pipes]
        else:
            random.shuffle(indices)
        
        imp = 0
        for i in indices:
            cand = list(current)
            if is_repair_mode:
                if cand[i] < self.n_diams - 1: cand[i] += 1
            else:
                if cand[i] > 0: cand[i] -= 1
            
            new_score_tuple = self.sim.evaluate(cand, "static", tolerance=0.0)
            new_score = new_score_tuple[0]
            
            if new_score < current_score:
                current = cand
                current_score = new_score
                imp += 1
                if is_repair_mode and new_score < 1e9: is_repair_mode = False 

        if verbose: 
            mode_str = "REPAIR" if current_score > 1e9 else "OPTIMIZE"
            print(f"   > LS [{mode_str}]: Done in {format_time(time.time()-start)}. Imp: {imp}. Final: {current_score/1e6:.4f} M$")
        return current

    def run(self, run_id, h_min):
        random.seed(time.time() + run_id)
        self.start_time = time.time()
        print(f"\n>>> Starting Run #{run_id + 1}...")

        # --- DIAGNOSTICS BLOCK (Відновлено) ---
        print("\n    [DIAGNOSTICS] Network Bounds:")
        
        # 1. MIN CONFIG (Найдешевша, усі труби = 0)
        min_ind = [0] * self.n_pipes
        min_cost, min_p_min, min_p_max = self.sim.get_stats(min_ind)
        print(f"    1. CHEAPEST (All Min): Cost={min_cost/1e6:.4f}M$ | P=[{min_p_min:.2f} .. {min_p_max:.2f}]m")
        
        # 2. MAX CONFIG (Найдорожча, усі труби = Max)
        max_ind = [self.n_diams - 1] * self.n_pipes
        max_cost, max_p_min, max_p_max = self.sim.get_stats(max_ind)
        print(f"    2. ROBUST   (All Max): Cost={max_cost/1e6:.4f}M$ | P=[{max_p_min:.2f} .. {max_p_max:.2f}]m")

        # Перевірка на можливість рішення
        if max_p_min < h_min:
             print(f"    [CRITICAL WARNING] Solution IMPOSSIBLE! Max pressure {max_p_min:.2f}m < Target {h_min}m")
        elif min_p_min >= h_min:
             print(f"    [INFO] Trivial solution! Cheapest pipes already satisfy pressure.")
        print("-" * 50)
        # --------------------------------------

        is_small_net = self.n_pipes < 150
        if is_small_net:
            ls_freq = 5; ls_limit = None; cache_enabled = False; tourn_size = 2; hyper_trigger = 10
            print(f"    [Mode] Aggressive (Small Network) | Init: Feasibility First")
        else:
            ls_freq = 10; ls_limit = 50; cache_enabled = True; tourn_size = 2; hyper_trigger = 15
            print(f"    [Mode] Deep Dive (Large Network) | Init: Feasibility First")

        self.toolbox.register("select", tools.selTournament, tournsize=tourn_size)

        pop = self._create_population()
        hof = tools.HallOfFame(1)
        
        for ind in pop: ind.fitness.values = self._cached_eval(ind, 0, cache_enabled)
        hof.update(pop)

        last_best_cost = hof[0].fitness.values[0]
        stagnation_counter = 0
        history = []

        for gen in range(self.n_gens):
            elapsed = time.time() - self.start_time
            if gen > 0:
                eta_str = format_time((elapsed / gen) * (self.n_gens - gen))
            else: eta_str = "..."

            # Anti-Stagnation
            current_best = hof[0].fitness.values[0]
            if abs(current_best - last_best_cost) < 1e-4:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_cost = current_best

            # Params
            base_mut_prob = MUTATION_START - (gen/self.n_gens)*(MUTATION_START - MUTATION_END)
            
            # Hyper Mutation Trigger
            triggered_hyper = False
            if stagnation_counter > hyper_trigger:
                print(f"    [Gen {gen}] ⚡ HYPER-MUTATION! (Stag: {stagnation_counter})")
                elite = self.toolbox.clone(hof[0])
                pop = self._apply_hyper_mutation(pop, strength=0.20)
                pop[0] = elite
                for ind in pop: 
                    if not ind.fitness.valid:
                        ind.fitness.values = self._cached_eval(ind, gen, cache_enabled)
                stagnation_counter = 0
                triggered_hyper = True
            
            elif stagnation_counter > 5:
                base_mut_prob = 0.8 

            self.toolbox.register("mutate", self._smart_mutation, indpb=base_mut_prob)

            if not triggered_hyper:
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))
                offspring[0] = self.toolbox.clone(hof[0])

                for child1, child2 in zip(offspring[1::2], offspring[2::2]):
                    if random.random() < 0.9: 
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values, child2.fitness.values
                
                for i in range(1, len(offspring)):
                    if random.random() < 0.4: 
                        self.toolbox.mutate(offspring[i])
                        del offspring[i].fitness.values
                
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                for ind in invalid_ind: ind.fitness.values = self._cached_eval(ind, gen, cache_enabled)
                pop[:] = offspring

            # Local Search
            if gen > 0 and gen % ls_freq == 0:
                cand = list(hof[0])
                improved = self.run_local_search(cand, limit_pipes=ls_limit, verbose=False)
                imp_ind = creator.Individual(improved)
                imp_ind.fitness.values = self._cached_eval(imp_ind, gen, cache_enabled)
                if imp_ind.fitness.values[0] < hof[0].fitness.values[0]:
                    hof.clear(); hof.update([imp_ind]); pop[0] = imp_ind

            # Update HOF
            best_pop = tools.selBest(pop, 1)[0]
            real_fit = self.sim.evaluate(best_pop, "static", gen, 0.0, self.n_gens)
            if real_fit[0] < hof[0].fitness.values[0]:
                nc = self.toolbox.clone(best_pop)
                nc.fitness.values = real_fit
                hof.clear(); hof.update([nc])

            history.append({'gen': gen, 'cost': hof[0].fitness.values[0]/1e6})
            
            if gen % 10 == 0 or gen == self.n_gens - 1:
                cost, min_p, max_p = self.sim.get_stats(hof[0])
                stag_info = f" [Stag:{stagnation_counter}]" if stagnation_counter > 3 else ""
                print(f"    [Gen {gen:3d}] Cost={cost/1e6:.2f}M$ | P=[{min_p:.2f}..{max_p:.2f}]m | Time={format_time(elapsed)} | ETA: {eta_str}{stag_info}")

        best_ind = hof[0]
        cost, min_p, max_p = self.sim.get_stats(best_ind)
        total_time = time.time() - self.start_time
        print(f"    >> Done in {format_time(total_time)}. Final: {cost/1e6:.4f}M$")
        return best_ind, cost, min_p, max_p, total_time, history