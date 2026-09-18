"""Кіки, що спираються на структуру мережі, а не на поточний шлях.

`LOOP_BALANCE` перерозподіляє діаметри всередині циклу, `ZERO_SUM`
міняє один крок угору на один униз зі збереженням вартості, `TRIM`
підрізає периферію. Спільне — вони дивляться на топологію й на вартість,
а не на те, де зараз просів тиск.
"""

import random

import networkx as nx

from .kick_registry import KickArgs, KickResult


class StructuralKicks:
    def loop_balancing_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        failed_pipes = args.failed_pipes
        current_round = args.current_round
        dyn_bonus = args.dyn_bonus

        LOOP_BALANCE_PIPE_TENURE = min(80, max(3, self.ctx.num_pipes // 5))
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return KickResult.failure()

        if not hasattr(self, '_cached_cycles'):
            try: self._cached_cycles = nx.cycle_basis(self.ctx.base_G_flow)
            except Exception: self._cached_cycles = []
        cycles = self._cached_cycles

        if not cycles: return KickResult.failure("No cycles found")

        max_allowed_drop = 1 + int(2 * T)

        best_drop_achieved = -1
        candidates = []
        random.shuffle(cycles)

        for cycle_nodes in cycles:
            cycle_indices = []
            full_cycle = cycle_nodes + [cycle_nodes[0]]
            for u, v in zip(full_cycle[:-1], full_cycle[1:]):
                if (u, v) in self.ctx.edge_to_pipe:
                    cycle_indices.append(self.ctx.edge_to_pipe[(u, v)])

            if not cycle_indices: continue

            restrict_pct = 0.10 + (0.30 * T)
            n_restrict = max(1, int(len(cycle_indices) * restrict_pct))

            for _ in range(3):
                chosen_pipes = random.sample(cycle_indices, n_restrict)

                if any((current_round - failed_pipes.get(p, -999)) < LOOP_BALANCE_PIPE_TENURE for p in chosen_pipes):
                    continue
                if any(indices[p] < 2 for p in chosen_pipes):
                    continue

                max_drop = min(max_allowed_drop, min(indices[p] for p in chosen_pipes))

                for drop in range(max_drop, 0, -1):
                    if drop < best_drop_achieved and n_restrict == 1: continue

                    kicked, locked = list(indices), set()

                    for p in chosen_pipes:
                        kicked[p] -= drop
                        locked.add(p)

                    healed_sol, is_feasible, boosts = self.ls.heal_network(kicked, locked)
                    if is_feasible:
                        test_squeezed = self.ls.gradient_squeeze(healed_sol, locked_pipes=locked, max_passes=2, quick_mode=True, dyn_bonus=dyn_bonus)
                        sq_cost, _, _, _ = self.ctx.get_cached_stats(test_squeezed)

                        candidates.append((sq_cost, healed_sol, locked,
                                           f"FLOW STEER: Cut {n_restrict} pipes in loop (-{drop}). Healed {boosts}x.",
                                           chosen_pipes[0]))

                        best_drop_achieved = max(best_drop_achieved, drop)
                        break

            if len(candidates) >= 5: break

        if not candidates: return KickResult.failure("FLOW STEER: Exhaustive search found no valid bypass.")
        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])
        return KickResult(chosen[1], chosen[2], chosen[3], failed_pipe=chosen[4])

    def zero_sum_shift_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return KickResult.failure()

        zs_tabu = args.zero_sum_tabu
        curr_round = args.current_round

        unit_losses = self.ctx.get_cached_heuristics(indices)
        kicked = list(indices)

        upgrades = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] < self.ctx.max_d_idx:
                c_up = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]+1] - self.ctx.costs_array[kicked[i]])
                upgrades.append((i, c_up, unit_losses[i] / max(c_up, 1.0)))
        upgrades.sort(key=lambda x: x[2], reverse=True)

        valid_upgrades = []
        for up in upgrades:
            if curr_round - zs_tabu.get(up[0], -999) > 15:
                valid_upgrades.append(up)

        downgrades = []
        for i in range(self.ctx.num_pipes):
            if kicked[i] > 0:
                c_down = self.ctx.lengths[i] * (self.ctx.costs_array[kicked[i]] - self.ctx.costs_array[kicked[i]-1])
                downgrades.append((i, c_down, c_down / max(unit_losses[i], 1e-5)))
        downgrades.sort(key=lambda x: x[2], reverse=True)

        best_sol, best_locked, best_cost = None, None, float('inf')

        search_pool_size = max(15, self.ctx.num_pipes // 10)

        tests_limit = 5 if T < 0.3 else 2

        base_down = max(1, self.ctx.num_pipes // 50)
        max_downgrades = base_down if T < 0.2 else (base_down * 3 if T < 0.5 else base_down * 6)

        valid_found = 0

        for up_idx, cost_invest, _ in valid_upgrades[:search_pool_size]:
            if valid_found >= tests_limit: break

            test_sol = list(indices)
            test_sol[up_idx] += 1
            locked_set = {up_idx}
            savings = 0
            down_count = 0

            for down_idx, cost_save, _ in downgrades:
                if down_idx == up_idx or down_idx in locked_set: continue

                test_sol[down_idx] -= 1
                locked_set.add(down_idx)
                savings += cost_save
                down_count += 1

                if savings > cost_invest:
                    healed_sol, ok, _ = self.ls.heal_network(test_sol, locked_set)
                    if ok:
                        c, _, _, _ = self.ctx.get_cached_stats(healed_sol)
                        if c < best_cost:
                            best_cost, best_sol, best_locked, best_up = c, healed_sol, locked_set, up_idx
                        valid_found += 1
                    break

                if down_count >= max_downgrades:
                    break

        if best_sol:
            mode_str = "1-to-1" if len(best_locked) == 2 else f"1-to-{len(best_locked)-1}"
            return KickResult(best_sol, best_locked, f"ZERO-SUM ({mode_str}): P{best_up} upgraded. Cost: {best_cost/1e6:.4f}M$")

        return KickResult.failure()

    def peripheral_trim_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        indices_copy, periphery_pipes = self.ctx.get_lazy_periphery(indices)
        if not periphery_pipes:
            return KickResult.failure("No targets")

        trim_pct = 0.02 + (0.13 * T)
        pipes_to_cut = max(1, int(len(periphery_pipes) * trim_pct))

        candidates = []
        combo_limit = min(len(periphery_pipes), max(10, self.ctx.num_pipes // 2))
        max_checks = min(25, max(5, self.ctx.num_pipes // 10))
        dyn_bonus = args.dyn_bonus

        for _ in range(max_checks):
            combo = random.sample(periphery_pipes[:combo_limit], pipes_to_cut)

            valid_combo = False
            for p in combo:
                if indices_copy[p] > 0:
                    valid_combo = True
                    break

            if not valid_combo:
                continue

            test_sol = list(indices_copy)
            locked = set()

            for p in combo:
                if test_sol[p] > 0:
                    test_sol[p] -= 1
                    locked.add(p)

            healed_sol, is_feasible, boosts = self.ls.heal_network(test_sol, locked)
            if is_feasible:
                h_c, h_p, _, _ = self.ctx.get_cached_stats(healed_sol)
                h_score = self.ctx.score(h_c, h_p, dyn_bonus)
                candidates.append((h_score, healed_sol, locked, boosts))

        if not candidates:
            return KickResult.failure("No valid trim candidates found or heal failed")

        candidates.sort(key=lambda x: x[0])
        chosen = random.choice(candidates[:3])

        return KickResult(chosen[1], chosen[2], f"TRIM: Shrunk {len(chosen[2])} peripheral pipes. Healed {chosen[3]}x.")
