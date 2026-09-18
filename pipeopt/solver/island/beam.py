"""Промінь: мутації, відбір, аварійні виходи.

Це та частина, яка на великій мережі з'їдає майже весь бюджет: замір
16.09.2026 дав 21 600 симуляцій на раунд на Балермі проти 200 на Ханої,
з яких на диверсифікацію не лишається нічого.

Аварійні гілки (`_emergency_respawn`, `_emergency_pool_diversity`,
`_check_mini_restart`) спрацьовують лише на малих мережах — там, де
промінь встигає вичерпатись.
"""

import itertools
import random

import numpy as np

try:
    from ..fast_math import fast_avg_hamming
    HAS_FAST_MATH = True
except ImportError:
    HAS_FAST_MATH = False

from ..kick_registry import (
    KickArgs,
)


class BeamSearch:
    """Покоління кандидатів, відбір променем, аварійні виходи."""

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
                if self._is_ghost_gain(c, swapped):
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
                    # Вікна відносної допустимості тут більше немає: гілка
                    # для рішень із легким порушенням тиску була недосяжна
                    # (`evaluate_candidate` відсікає їх раніше), а замір на
                    # трьох мережах показав, що її оживлення не дає виграшу.
                    # Кандидати, що сюди доходять, допустимі за побудовою —
                    # це зафіксовано в tests/test_epsilon_window.py.
                    if st.is_feasible(self.ctx.h_min):
                        score = self.ctx.score(real_cost, st.p_min, self.base_dyn_bonus)

                        if real_cost < self.run_best_cost:
                            if self._is_ghost_gain(real_cost, refined_sol):
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

        # Кіки повертають `KickResult` (§13.1), а не кортеж. Перевірка
        # `isinstance(res, tuple)`, що стояла тут до цього, була завжди
        # хибною, тому обидва рятувальні збурення виконувались і їхній
        # результат викидався, а пул наповнював найгрубіший запасний
        # варіант нижче. Тест: test_emergency_respawn.py.
        res1 = self.kicker.smart_perturbation_kick(
            self.run_best_sol, KickArgs(T=1.0))
        forced_rescue = None if res1.failed else res1.solution

        if forced_rescue is not None:
            st = self.ctx.get_cached_stats(forced_rescue)
            if st.is_feasible(self.ctx.h_min):
                self.pool.active_pool.append((st.cost - 1e6, st.cost, forced_rescue))
                self.stagnation_counter = 0

        if not self.pool.active_pool:
            res2 = self.kicker.smart_perturbation_kick(
                self.run_best_sol, KickArgs(T=0.6))
            healed_seed = None if res2.failed else res2.solution

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
