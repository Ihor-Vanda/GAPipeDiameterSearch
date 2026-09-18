"""Кіки, що ламають частину рішення й відбудовують її наново.

Чотири стратегії з самоадаптивним розміром кроку (JADE): вони збурюють
частку мережі, лікують результат і повідомляють `used_pct`, з якого
`AdaptiveRates` вчить розмір наступного збурення. Саме через це вони
зібрані разом — у них спільний контракт із механізмом самонавчання.
"""

import random

import networkx as nx

from .kick_registry import KickArgs, KickResult


class RuinKicks:
    def smart_perturbation_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        base_mu = args.mu_perturb_pct
        dynamic_mu = base_mu + (0.15 * T)
        dynamic_sigma = 0.02 + (0.05 * T)
        target_pct = random.gauss(dynamic_mu, dynamic_sigma)
        target_pct = max(0.01, min(0.30, target_pct))

        max_perturb = args.max_perturb or self.ctx.num_pipes
        n_perturb = min(max_perturb, max(1, int(self.ctx.num_pipes * target_pct)))

        kicked, locked = list(indices), set()
        candidates = [i for i in range(self.ctx.num_pipes) if 0 < kicked[i] < self.ctx.max_d_idx]
        if len(candidates) < n_perturb: candidates = list(range(self.ctx.num_pipes))

        chosen = random.sample(candidates, min(n_perturb, len(candidates)))

        max_delta = 2 if T > 0.6 else 1

        for p_idx in chosen:
            if max_delta > 1: delta = random.choices([-2, -1, 1, 2], weights=[1, 3, 3, 1])[0]
            else: delta = random.choice([-1, 1])

            new_val = max(0, min(self.ctx.max_d_idx, kicked[p_idx] + delta))
            kicked[p_idx] = new_val
            if delta > 0: locked.add(p_idx)

        healed, ok, boosts = self.ls.heal_network(kicked, locked)
        if not ok: return KickResult.failure()
        if list(healed) == list(indices):
            return KickResult.failure("SMART-PERTURB: shift produced no change")
        return KickResult(healed, locked, f"SMART-PERTURB: Shifted {len(chosen)} pipes (Size: {target_pct:.1%}). Healed {boosts}x.", used_pct=target_pct)

    def ruin_and_recreate_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return KickResult.failure()

        ruin_center = crit_node if T < 0.8 else random.choice(list(self.ctx.base_G_flow.nodes()))

        base_mu = args.mu_ruin_pct
        dynamic_sigma = 0.01 + (0.04 * T)
        target_pct = random.gauss(base_mu, dynamic_sigma)
        target_pct = max(0.01, min(0.25, target_pct))

        target_pipes = max(3, int(self.ctx.num_pipes * target_pct))

        n = self.ctx.num_pipes
        cutoff = 3 if n < 50 else (4 if n < 200 else (6 if n < 1000 else 8))

        pipe_with_dist, seen_pipes = [], set()

        try:
            for node, dist in nx.single_source_shortest_path_length(self.ctx.base_G_flow, ruin_center, cutoff=cutoff).items():
                for p in self.ctx.node_to_pipes.get(node, []):
                    if p not in seen_pipes:
                        seen_pipes.add(p)
                        pipe_with_dist.append((dist, p))
        except Exception: pass

        if not pipe_with_dist: return KickResult.failure()
        pipe_with_dist.sort(key=lambda x: x[0])
        cluster_pipes = set(p for _, p in pipe_with_dist[:target_pipes])

        kicked, ruined_count = list(indices), 0
        max_catalog_drop = max(1, self.ctx.max_d_idx // 3)
        temp_drop = 1 + int(2 * T)
        max_drop = min(temp_drop, max_catalog_drop)

        for p in cluster_pipes:
            if kicked[p] > 0:
                kicked[p] = max(0, kicked[p] - random.randint(1, max_drop))
                ruined_count += 1

        if ruined_count == 0: return KickResult.failure()

        healed_sol, is_feas, boosts = self.ls.heal_network(kicked, set())
        if not is_feas: return KickResult.failure()

        # Лікування йде **без замків**, тому воно має право підняти назад
        # саме ті труби, які кік щойно опустив, — і рішення повертається
        # у точку старту. Замір на Ханої (120 тис., зерна 42/1/7): з 150
        # викликів R&R таких «кругових» — 91, 97 і 78, тобто близько 60 %.
        # Кожен із них коштував лікування, поліровки й класифікації, а
        # UCB1 отримував за нього нагороду як за успіх.
        if list(healed_sol) == list(indices):
            return KickResult.failure("R&R: heal restored the original solution")

        return KickResult(healed_sol, set(), f"R&R: Downgraded {ruined_count} pipes (Size: {target_pct:.1%}). Healed {boosts}x.", used_pct=target_pct)

    def basin_escape(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        global_archive = args.global_archive
        if not global_archive or len(global_archive) < 2:
            return KickResult.failure("Archive too small")

        num_pipes = len(indices)
        min_dist_threshold = max(5, int(num_pipes * 0.05))

        target_sol = None
        max_dist = -1

        for arch_cost, arch_sol in global_archive:
            dist = sum(1 for a, b in zip(indices, arch_sol) if a != b)

            if dist > max_dist and dist >= min_dist_threshold:
                max_dist = dist
                target_sol = arch_sol

        if target_sol is None:
            return KickResult.failure("Archive degenerated. No distinct basins found.")

        diff_pipes = [i for i in range(num_pipes) if indices[i] != target_sol[i]]

        base_mu = args.mu_escape_pct
        target_pct = max(0.05, min(0.60, random.gauss(base_mu + (0.20 * T), 0.05 + (0.05 * T))))

        n_replace = max(1, int(len(diff_pipes) * target_pct))
        replace_pipes = random.sample(diff_pipes, n_replace)
        kicked = list(indices)
        for p in replace_pipes: kicked[p] = target_sol[p]

        healed, ok, boosts = self.ls.heal_network(kicked, set())
        if not ok:
            return KickResult.failure("Heal failed after transplant")

        log_msg = f"BASIN-ESCAPE: Transplanted {n_replace} pipes (Size: {target_pct:.1%}). Healed {boosts}x."
        return KickResult(healed, set(), log_msg, used_pct=target_pct)

    def spatial_perturb_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        num_pipes = len(indices)

        base_mu = args.mu_spatial_pct
        dynamic_sigma = 0.02 + (0.10 * T)
        target_pct = max(0.02, min(0.60, random.gauss(base_mu, dynamic_sigma)))

        target_size = max(2, int(num_pipes * target_pct))

        max_p = args.max_perturb or int(num_pipes * 0.50)
        target_size = min(target_size, max_p)

        physical_G = nx.Graph(self.ctx.base_G_flow)

        local_pipes = set()

        for _ in range(5):
            epicenter_node = random.choice(list(physical_G.nodes()))

            radius = 1
            while len(local_pipes) < target_size and radius < 20:
                subgraph = nx.ego_graph(physical_G, epicenter_node, radius=radius, undirected=True)

                for u, v in subgraph.edges():
                    key = (u, v)
                    if key in self.ctx.edge_to_pipe:
                        local_pipes.add(self.ctx.edge_to_pipe[key])
                    elif (v, u) in self.ctx.edge_to_pipe:
                        local_pipes.add(self.ctx.edge_to_pipe[(v, u)])

                radius += 1

                if radius > 1 and len(subgraph.nodes()) == len(nx.ego_graph(physical_G, epicenter_node, radius=radius-1).nodes()):
                    break

            if len(local_pipes) >= target_size:
                break

        if len(local_pipes) < target_size:
            local_pipes = set(random.sample(range(num_pipes), target_size))

        local_pipes = list(local_pipes)[:target_size]

        kicked = list(indices)
        for p in local_pipes:
            shift = random.choice([-2, -1, 1, 2])
            kicked[p] = max(1, min(self.ctx.max_d_idx - 1, kicked[p] + shift))

        healed, ok, boosts = self.ls.heal_network(kicked, set())
        if not ok:
            return KickResult.failure("Heal failed after spatial perturb")

        if list(healed) == list(indices):
            return KickResult.failure("SPATIAL-PERTURB: heal restored the original")

        log_msg = f"SPATIAL-PERTURB: Shifted {len(local_pipes)} pipes (Size: {target_pct:.1%}). Healed {boosts}x."

        return KickResult(healed, set(), log_msg, used_pct=target_pct)
