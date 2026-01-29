# ga_optimizer.py
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

        # DEAP setup
        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        if hasattr(creator, "Individual"): del creator.Individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("attr_int", random.randint, 0, self.n_diams-1)

    def _smart_mutation(self, individual, indpb, stagnation_factor=1.0):
        effective_pb = indpb * stagnation_factor
        
        for i in range(len(individual)):
            if random.random() < effective_pb:
                if random.random() < 0.9: 
                    step = random.randint(1, 2)
                    change = step if random.random() < 0.5 else -step
                    new_val = individual[i] + change
                    if new_val < 0: new_val = 0
                    elif new_val > self.n_diams - 1: new_val = self.n_diams - 1
                    individual[i] = new_val
                else:
                    individual[i] = random.randint(0, self.n_diams - 1)
        return individual,

    def _create_population(self):
        pop = []
        for _ in range(int(self.pop_size * 0.1)):
            pop.append(creator.Individual([self.n_diams - 1] * self.n_pipes))
        mid_idx = self.n_diams // 2
        for _ in range(int(self.pop_size * 0.4)):
            pop.append(creator.Individual([mid_idx] * self.n_pipes))
        remaining = self.pop_size - len(pop)
        for _ in range(remaining):
            pop.append(creator.Individual([random.randint(0, self.n_diams-1) for _ in range(self.n_pipes)]))
        return pop

    def _cached_eval(self, individual, gen, cache_enabled=True):
        if gen < self.n_gens * 0.75:
            progress = gen / (self.n_gens * 0.75)
            tol = max(0.0, EPSILON_START - progress * (EPSILON_START - EPSILON_END))
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
        current_cost = self.sim.evaluate(current, "static", tolerance=0.0)[0]
        
        indices = list(range(self.n_pipes))
        if limit_pipes and limit_pipes < self.n_pipes:
            indices = random.sample(indices, k=limit_pipes)
        
        if verbose: print(f"   > Local Search ({len(indices)} pipes)...")
        
        imp = 0
        for i in indices:
            if current[i] > 0:
                cand = list(current)
                cand[i] -= 1
                new_cost = self.sim.evaluate(cand, "static", tolerance=0.0)[0]
                
                if new_cost < current_cost:
                    current = cand
                    current_cost = new_cost
                    imp += 1
        
        if verbose: print(f"   > Done in {format_time(time.time()-start)}. Imp: {imp}. Final: {current_cost/1e6:.4f} M$")
        return current

    def run(self, run_id, h_min):
        random.seed(time.time() + run_id)
        self.start_time = time.time()
        print(f"\n>>> Starting Run #{run_id + 1}...")

        is_small_net = self.n_pipes < 100
        
        if is_small_net:
            ls_freq = 5
            ls_limit = None
            cache_enabled = False
            tourn_size = 2
            print(f"    [Mode] Aggressive (Small Network) | TournSize=2 | No Cache")
        else:
            ls_freq = 20
            ls_limit = 20
            cache_enabled = True
            tourn_size = 3
            print(f"    [Mode] Eco (Large Network) | TournSize=3 | Cache ON")

        self.toolbox.register("select", tools.selTournament, tournsize=tourn_size)

        max_ind = [self.n_diams - 1] * self.n_pipes
        _, max_p = self.sim.get_stats(max_ind)
        print(f"    [INFO] Feasibility Check: P={max_p:.2f} m")
        if max_p < h_min:
            print(f"    [CRITICAL] Target {h_min}m IMPOSSIBLE. Max is {max_p:.2f}m")

        pop = self._create_population()
        hof = tools.HallOfFame(1)
        history = []

        for ind in pop: ind.fitness.values = self._cached_eval(ind, 0, cache_enabled)
        hof.update(pop)

        last_best_cost = hof[0].fitness.values[0]
        stagnation_counter = 0

        for gen in range(self.n_gens):
            base_mut_prob = MUTATION_START - (gen/self.n_gens)*(MUTATION_START - MUTATION_END)
            
            current_best_cost = hof[0].fitness.values[0]
            if abs(current_best_cost - last_best_cost) < 1e-3:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_cost = current_best_cost

            stag_factor = 3.0 if stagnation_counter > 10 else 1.0
            
            self.toolbox.register("mutate", self._smart_mutation, indpb=base_mut_prob, stagnation_factor=stag_factor)

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            offspring[0] = self.toolbox.clone(hof[0]) 

            for child1, child2 in zip(offspring[1::2], offspring[2::2]):
                if random.random() < 0.8: 
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            
            for i in range(1, len(offspring)):
                if random.random() < 0.35: 
                    self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind: ind.fitness.values = self._cached_eval(ind, gen, cache_enabled)
            
            pop[:] = offspring

            if gen > 0 and gen % ls_freq == 0:
                candidate = list(hof[0])
                improved_cand = self.run_local_search(candidate, limit_pipes=ls_limit, verbose=False)
                
                imp_ind = creator.Individual(improved_cand)
                imp_ind.fitness.values = self._cached_eval(imp_ind, gen, cache_enabled)
                
                if imp_ind.fitness.values[0] < hof[0].fitness.values[0]:
                    hof.clear()
                    hof.update([imp_ind])
                    pop[0] = imp_ind

            best_in_pop = tools.selBest(pop, 1)[0]
            real_fit = self.sim.evaluate(best_in_pop, "static", gen, 0.0, self.n_gens)
            if real_fit[0] < hof[0].fitness.values[0]:
                nc = self.toolbox.clone(best_in_pop)
                nc.fitness.values = real_fit
                hof.clear(); hof.update([nc])

            history.append({'gen': gen, 'cost': hof[0].fitness.values[0]/1e6})
            
            if gen % 10 == 0 or gen == self.n_gens - 1:
                cost, p = self.sim.get_stats(hof[0])
                elapsed = time.time() - self.start_time
                stag_msg = " [STAGNATION]" if stagnation_counter > 10 else ""
                print(f"    [Gen {gen:3d}] Cost={cost/1e6:.4f}M$ | P={p:.2f}m | Time={format_time(elapsed)}{stag_msg}")

        best_ind = hof[0]
        cost, p = self.sim.get_stats(best_ind)
        total_time = time.time() - self.start_time
        print(f"    >> Done in {format_time(total_time)}. Final: {cost/1e6:.4f}M$")
        return best_ind, cost, p, total_time, history