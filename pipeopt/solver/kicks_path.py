"""Кіки, що працюють із домінантним шляхом до критичного вузла.

Спільне в них — точка прикладання: усі троє спершу питають `ctx`, який
вузол зараз найгірший за тиском, і будують хід уздовж шляху до нього.
Тому вони й починаються однаково, і ламаються однаково, коли шляху немає.
"""

import random

import networkx as nx
import numpy as np

from .kick_registry import KickArgs, KickResult


class PathKicks:
    def forcing_hand_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR":
            return KickResult.failure("No critical node")

        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes:
            return KickResult.failure("No dominant path")

        kicked, locked = list(indices), set()

        aggressiveness = 0.05 + (0.35 * T)
        limit = max(1, int(len(path_pipes) * aggressiveness))

        unit_losses = self.ctx.get_cached_heuristics(indices)
        worst_pipes = sorted(path_pipes, key=lambda i: unit_losses[i], reverse=True)[:limit]

        for idx in worst_pipes:
            if kicked[idx] < self.ctx.max_d_idx:
                kicked[idx] += 1
                locked.add(idx)

        if not locked:
            return KickResult.failure("Every pipe on the critical path is already at the maximum")
        return KickResult(kicked, locked, f"SHOCK: Upgraded {len(locked)} worst pipes on Critical Path (T={T:.2f})")

    def upstream_bottleneck_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        failed_pipes = args.failed_pipes
        boosted_pipes = args.boosted_pipes
        current_round = args.current_round
        tenure = 5

        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR":
            return KickResult.failure("No critical node")

        path_pipes, _ = self.ctx.get_dominant_path(indices, crit_node)
        if not path_pipes:
            return KickResult.failure("No dominant path")

        kicked, locked = list(indices), set()

        valid_taper_violations = []
        for i in range(len(path_pipes) - 1, 0, -1):
            curr_p, prev_p = path_pipes[i], path_pipes[i-1]

            if curr_p in failed_pipes and (current_round - failed_pipes[curr_p]) < 10:
                continue

            if indices[curr_p] < indices[prev_p]:
                if kicked[curr_p] < self.ctx.max_d_idx:
                    valid_taper_violations.append(curr_p)

        if valid_taper_violations:
            target_pipe = None
            for p in valid_taper_violations:
                if p not in boosted_pipes or (current_round - boosted_pipes[p]) >= tenure:
                    target_pipe = p
                    break

            if target_pipe is None:
                target_pipe = valid_taper_violations[0]

            kicked[target_pipe] += 1
            locked.add(target_pipe)
            return KickResult(kicked, locked, f"BOTTLENECK: Boosted Pipe {target_pipe} (Taper Violation)", failed_pipe=target_pipe)

        unit_losses = self.ctx.get_cached_heuristics(indices)
        search_depth = max(1, int(len(path_pipes) * (0.3 + 0.4 * T)))

        base_candidates = [(idx, unit_losses[idx]) for idx in path_pipes[:search_depth]
                           if indices[idx] < self.ctx.max_d_idx
                           and not (idx in failed_pipes and (current_round - failed_pipes[idx]) < 10)]

        if not base_candidates:
            return KickResult.failure("No boostable pipe left on the critical path")

        candidates = [c for c in base_candidates
                      if c[0] not in boosted_pipes or (current_round - boosted_pipes[c[0]]) >= tenure]

        if not candidates:
            candidates = base_candidates

        boost_pct = 0.05 + (0.15 * T)
        n_boost = max(1, int(len(candidates) * boost_pct))

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_pipes = [x[0] for x in candidates[:n_boost]]

        for p in best_pipes:
            kicked[p] += 1
            locked.add(p)

        return KickResult(kicked, locked, f"BOTTLENECK: Boosted {len(best_pipes)} Max-Loss Pipes.", failed_pipe=best_pipes[0])

    def topological_inversion_kick(self, indices, args: KickArgs) -> KickResult:
        T = args.T
        tabu_set = args.tabu_set
        _, _, _, crit_node = self.ctx.get_cached_stats(indices)
        if not crit_node or crit_node == "ERR": return KickResult.failure()

        dom_pipes, dom_nodes = self.ctx.get_dominant_path(indices, crit_node)
        if not dom_nodes: return KickResult.failure()

        source = dom_nodes[0]
        dom_edges = set(tuple(sorted((u, v))) for u, v in zip(dom_nodes[:-1], dom_nodes[1:]))
        target_capacity_idx = int(np.mean([indices[p] for p in dom_pipes])) if dom_pipes else self.ctx.max_d_idx

        temp_G = self.ctx.base_G_flow.copy()
        for u, v in temp_G.edges():
            idx = self.ctx.edge_to_pipe[(u, v)]
            temp_G[u][v]['weight'] = 100.0 / (indices[idx] + 1)

        candidates = []
        for _ in range(5):
            try:
                path = nx.shortest_path(temp_G, source, crit_node, weight='weight')
                if path != dom_nodes: candidates.append(path)
                for u, v in zip(path[:-1], path[1:]): temp_G[u][v]['weight'] *= 5.0
            except nx.NetworkXNoPath: break

        scored_paths = []
        for p_nodes in candidates:
            p_indices, overlap, length = [], 0, 0
            for u, v in zip(p_nodes[:-1], p_nodes[1:]):
                edge_key = tuple(sorted((u, v)))
                if edge_key in dom_edges: overlap += 1
                if (u, v) in self.ctx.edge_to_pipe:
                    p_indices.append(self.ctx.edge_to_pipe[(u, v)])
                    length += 1
            if length == 0: continue
            signature = tuple(sorted(p_indices))
            if signature in tabu_set: continue
            scored_paths.append({'indices': p_indices, 'overlap': overlap / length, 'signature': signature})

        scored_paths.sort(key=lambda x: x['overlap'])
        if not scored_paths: return KickResult.failure("All paths tabu")

        best_alt = scored_paths[0]
        kicked, locked = list(indices), set()

        aggressiveness = 0.10 + (0.50 * T)
        max_pipes = max(2, int(len(best_alt['indices']) * aggressiveness))

        indices_to_change = best_alt['indices']
        if len(indices_to_change) > max_pipes:
             indices_to_change = random.sample(indices_to_change, max_pipes)

        for idx in indices_to_change:
            force_idx = min(target_capacity_idx, self.ctx.max_d_idx)

            if kicked[idx] < force_idx:
                kicked[idx] = force_idx
                locked.add(idx)
            elif kicked[idx] == force_idx:
                locked.add(idx)

        # Труби, ширші за магістраль, не змінюються жодною з гілок вище,
        # і якщо вибірка складається лише з них, кік повертає вхідне
        # рішення. Етап 13b встановив контракт «порожній кік повідомляє
        # невдачу», але ця гілка його порушувала: на ланцюговій тестовій
        # мережі альтернативних шляхів не буває, тому вона не
        # перевірялась. Замір на Ханої (120 тис., зерно 42): 157 викликів
        # TOPO_INV, з них 2 без жодної зміни — і UCB1 нагороджував
        # стратегію за обидва.
        if kicked == list(indices):
            return KickResult.failure(
                "TOPO-INV: alt path already at trunk capacity")

        return KickResult(kicked, locked, f"TOPO-INV: Boosted Alt Path (Overlap {best_alt['overlap']:.2f}, T={T:.2f})", path_signature=best_alt['signature'])
