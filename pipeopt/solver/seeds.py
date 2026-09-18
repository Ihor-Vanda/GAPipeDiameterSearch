"""Початкові рішення: з чого починає кожен острів.

Винесено з `orchestrator.py` на етапі 16. `SeedFactory` не має спільного
стану ні з воркером, ні з солвером — вона будувала рішення й лежала в
одному файлі з ними лише історично.
"""

import bisect
import math
import random


class SeedFactory:
    def __init__(self, ctx, local_search):
        """`n_workers` і `beam_width` прибрані на етапі 19.

        Вони зберігались у полях і не читались **жодним** методом, а три
        місця виклику передавали їх щоразу. Знайдено AST-аналізом полів,
        які пишуться й не читаються.
        """
        self.ctx = ctx
        self.ls = local_search

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
