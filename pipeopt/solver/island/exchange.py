"""Обмін між островами: межа, щеплення, порятунок, міграція.

Усе, що вимикається прапорцем `--sequential`. Еталон іде саме з ним,
тому жоден рядок цього модуля в еталонному прогоні не виконується —
перевіряють його `tests/test_island_cooperation.py` і
`reference_run.py --islands`.

Спільний словник живе в чужому процесі, тому будь-який запис у нього
може впасти. `_publish` і `_read_shared` — єдине місце, де це
обробляється; решта модуля ходить тільки через них.
"""

import random




class IslandExchange:
    """Усе, що острів каже сусідам і чує від них."""

    def _publish(self, shared_progress, key, value):
        """Записати у спільний словник островів; повернути, чи вдалося.

        `shared_progress` — це `Manager().dict()`, який живе в окремому
        процесі. Будь-який запис у нього може впасти
        (`BrokenPipeError`, `EOFError`, `ConnectionResetError`) з причин,
        що не мають стосунку до пошуку: менеджер завершився, зв'язок
        обірвався, головний процес зупиняється.

        Обрив зв'язку — прикрість, а не причина втрачати результат
        багатогодинного прогону. Два з п'яти місць обміну були захищені
        саме так, а три — ні, і падіння в них убивало острів разом із
        усім, що він знайшов; оркестратор бачив лише
        `[Error] Worker N crashed` (знайдено 16.09.2026). Тепер правило
        одне на всі: невдача обміну йде в лог, пошук триває.
        """
        if shared_progress is None:
            return False
        try:
            shared_progress[key] = value
            return True
        except Exception as e:
            self.ctx.log(f"   [IPC] Не вдалось записати {key!r}: {e}")
            return False

    def _read_shared(self, shared_progress, key, default=None):
        """Прочитати зі спільного словника; `default`, якщо не вдалося.

        Читання падає з тих самих причин, що й запис, — і так само не має
        зупиняти пошук.
        """
        if shared_progress is None:
            return default
        try:
            return shared_progress.get(key, default)
        except Exception as e:
            self.ctx.log(f"   [IPC] Не вдалось прочитати {key!r}: {e}")
            return default

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

    #: Наскільки глобальний рекорд має випереджати власний, щоб мігрувати.
    #: Менша різниця не варта того, щоб ламати власну траєкторію.
    MIGRATION_LAG_FRACTION = 1.005
    #: Межі температури між трьома політиками міграції.
    SOFT_MIGRATION_T = 0.3
    HYBRID_MIGRATION_T = 0.7

    def _maybe_migrate(self, gb, shared_progress):
        """Синхронізація з сусідами, якщо настав час і є з ким.

        До етапу 16 ці 63 рядки жили прямо в тілі `run()` — це був
        найбільший і найглибше вкладений шматок циклу (шість рівнів
        відступу). Пункт §4.4 плану вимагав винести його ще на етапі 4;
        тут він нарешті винесений.

        Що саме вирішується тут: чи взагалі дивитись на сусідів (інтервал
        за симуляціями), чи варто (власний рекорд відстає), і **як** —
        за трьома політиками, які обирає температура застою.
        """
        if shared_progress is None:
            return
        curr_sims = self.ctx.sim_count
        if (curr_sims - self._last_migration_sims) < self._migration_interval_sims:
            return
        self._last_migration_sims = curr_sims

        archive = self._read_shared(shared_progress, 'global_archive')
        if archive is not None:
            self.global_archive = archive

        if gb is None:
            return
        gb_cost, gb_sol = gb
        if self.run_best_cost <= gb_cost * self.MIGRATION_LAG_FRACTION:
            return

        T = self.temperature
        if T < self.SOFT_MIGRATION_T:
            self._migrate_soft(gb_cost, gb_sol, shared_progress)
        elif T < self.HYBRID_MIGRATION_T:
            self._migrate_hybrid(gb_sol, T)
        else:
            self.ctx.log(f" 🛡️ [EXPLORATION SHIELD] Ignored global best. "
                         f"Deep Random Walk (T={T:.2f}).")

    def _pick_peer_solution(self, gb_cost, shared_progress):
        """Рішення випадкового сусіда, яке відрізняється від глобального.

        Потрібне як джерело шуму: гібрид лише з глобальним рекордом
        стягує всі острови в одну точку. `abs(...) > 0.1` відсіює сусіда,
        який сам і є поточним глобальним рекордом.
        """
        peers = [i for i in range(self.n_workers) if i != self.worker_id]
        random.shuffle(peers)
        for peer_id in peers:
            peer_data = shared_progress.get(f'best_sol_{peer_id}')
            if peer_data and abs(peer_data[0] - gb_cost) > 0.1:
                return peer_data[1]
        return None

    def _migrate_soft(self, gb_cost, gb_sol, shared_progress):
        """Холодний острів: підкинути гібрид у пул, не чіпаючи власний рекорд.

        Рекорд лишається своїм — мігрант іде в пул як ще один кандидат.
        Тому ця політика найбезпечніша і застосовується, поки застій малий.
        """
        hybrid_sol = self._spatial_crossover(self.run_best_sol, gb_sol, T=0.25)

        peer_sol = self._pick_peer_solution(gb_cost, shared_progress)
        noise_msg = ""
        if peer_sol:
            hybrid_sol = self._spatial_crossover(hybrid_sol, peer_sol, T=0.05)
            noise_msg = " + Peer Noise"

        _, p, _, _ = self.ctx.get_cached_stats(hybrid_sol)
        if p < self.ctx.h_min:
            hybrid_sol, ok, _ = self.ls.heal_network(hybrid_sol, set())
            if not ok:
                hybrid_sol = list(gb_sol)

        h_c, _, _, _ = self.ctx.get_cached_stats(hybrid_sol)

        self.pool.active_pool.insert(0, (h_c, h_c, hybrid_sol))
        self.stagnation_counter = 0
        self.ipc_immunity = 50
        self.ctx.log(f"    📡 [MIGRATION] Soft injection of Global Best "
                     f"hybrid{noise_msg} ({h_c/1e6:.4f}M$)")

    def _migrate_hybrid(self, gb_sol, T):
        """Теплий острів: гібрид може стати новим власним рекордом.

        УВАГА: перевірки тиску тут немає й не було — гібрид приймається
        **лише за вартістю**. Поведінка збережена свідомо (§2.7 ревізії);
        `FINAL GUARD` у кінці `run()` недопустиме рішення все одно лікує.
        Виправляти це = змінювати правило прийняття, тобто окреме
        рішення з власним A/B, а не частина розбору.
        """
        hybrid_sol = self._spatial_crossover(self.run_best_sol, gb_sol, T)

        c, p, _, _ = self.ctx.get_cached_stats(hybrid_sol)
        if p < self.ctx.h_min:
            hybrid_sol, ok, _ = self.ls.heal_network(hybrid_sol, set())
            if ok:
                c, p, _, _ = self.ctx.get_cached_stats(hybrid_sol)

        if c < self.run_best_cost:
            self.ctx.log(f" 🧬 [HYBRID SUCCESS] Created new hybrid solution: "
                         f"{c/1e6:.4f}M$")
            self.run_best_cost = c
            self.run_best_sol = hybrid_sol
            self.pool.active_pool.insert(0, (c, c, hybrid_sol))
            self.stagnation_counter = 0
        else:
            self.ctx.log(" 🧬 [HYBRID FAILED] Hybrid was too expensive. "
                         "Drift continues.")

    def _spatial_crossover(self, base_sol, donor_sol, T):
        import random
        import networkx as nx

        if not hasattr(self.ctx, 'base_G_flow'):
            return [donor_sol[i] if random.random() < 0.5 else base_sol[i] for i in range(self.ctx.num_pipes)]

        G = self.ctx.base_G_flow
        radius = 2 + int(6.0 * T)

        epicenter = random.choice(list(G.nodes()))
        local_nodes = set(nx.single_source_shortest_path_length(G, epicenter, cutoff=radius).keys())

        # Номер труби живе в `ctx.edge_to_pipe`, а не на ребрі: `base_G_flow`
        # будується як `add_edge(u, v)` без атрибутів (див. SolverContext).
        #
        # Раніше тут було два розбори: гілка для мультиграфа брала ключ
        # ребра, а гілка для звичайного графа шукала ціле число серед
        # `data.values()`. `base_G_flow` — **звичайний граф із порожніми
        # атрибутами**, тому друга гілка не знаходила жодної труби ніколи,
        # `local_pipes` лишався порожнім, і кожне схрещування йшло у
        # запасний варіант — випадкові 15 % мережі. Тобто «просторове»
        # схрещування було випадковим, а обчислений радіус — мертвим
        # кодом. Знайдено 15.09 разом із петльовою тестовою мережею;
        # послідовний еталон цього не бачить, бо схрещування живе в
        # міграції, а вона вимикається разом з островами.
        local_pipes = {self.ctx.edge_to_pipe[(u, v)]
                       for u, v in G.edges()
                       if (u in local_nodes or v in local_nodes)
                       and (u, v) in self.ctx.edge_to_pipe}

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

    def _update_global_best(self, shared_progress):
        if self.run_best_cost < self.last_published_cost:
            published = self._publish(
                shared_progress, f'best_sol_{self.worker_id}',
                (self.run_best_cost, list(self.run_best_sol)))
            # `last_published_cost` рухається лише після вдалого запису:
            # інакше один обрив зв'язку назавжди приховав би рекорд від
            # сусідів — сам воркер вважав би його вже опублікованим.
            if published or shared_progress is None:
                self.last_published_cost = self.run_best_cost

        if self.run_best_cost < self.global_best_cost:
            self.global_best_cost = self.run_best_cost
            current_gb = self._read_shared(shared_progress, 'global_best',
                                           (float('inf'), []))
            if shared_progress is not None and self.run_best_cost < current_gb[0]:
                self._publish(shared_progress, 'global_best',
                              (self.run_best_cost, list(self.run_best_sol)))
