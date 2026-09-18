"""Диспетчер кіків: обрати стратегію, вдарити, вирішити долю.

Сам удар живе в `kicks_*.py`; тут — сценарій навколо нього: вибір цілі,
збірка аргументів, лікування, поліровка й розв'язка (рекорд, втеча, пул,
остання спроба, відмова).

До етапу 17 це був один метод на 238 рядків із CC 79 — найгірший у
проєкті. Порядок обчислень збережено дослівно: `is_ghost_solution`
виконує симуляції, тому обчислювати її завчасно означало б змінити їх
кількість, а з нею й еталон.
"""

import random


from ..acceptance import Verdict
from ..kick_registry import (
    FALLBACK_KICK,
    KICK_REGISTRY,
    KickArgs,
    pool_for,
)
from ..strategy_selector import StrategySelector
from .state import AppliedKick



class Diversification:
    """Сценарій одного кіка: вибір, удар, розв'язка."""

    #: Температура, від якої б'ємо по найдальшому рішенню пулу.
    FAR_TARGET_T = 0.8
    #: Температура, від якої ціль обирається випадково.
    RANDOM_TARGET_T = 0.4
    #: Стратегії, які збурюють мережу цілком — їм блокування труб зайві.
    UNLOCKED_STRATEGIES = frozenset(("SMART_PERTURB", "SPATIAL_PERTURB",
                                    "RUIN_RECREATE"))
    #: Скільки рішень має бути в архіві, щоб консенсус щось означав.
    CONSENSUS_MIN_ARCHIVE = 3
    #: Частка прогону, після якої заморозка консенсусу вимикається.
    CONSENSUS_OFF_RATIO = 0.88
    #: Наскільки score гіршає при прийнятті «за рівнем води».
    POOL_SCORE_PENALTY = 1.05
    #: Те саме для останньої спроби.
    HAIL_MARY_SCORE_PENALTY = 1.20

    def _pick_kick_target(self, real_T, round_idx):
        """Із якого рішення бити.

        Три режими за температурою: при глибокому застої беремо
        **найдальше** від рекорда (аби вирватись), при середньому —
        випадкове з пулу, при малому — чергуємо рекорд, голову пулу й
        випадкове, щоб не довбати одну точку три раунди поспіль.
        """
        pool = self.pool.active_pool
        if real_T >= self.FAR_TARGET_T and pool:
            feasible = [x for x in pool
                        if self.ctx.get_cached_stats(x[2])[1] >= self.ctx.h_min]
            source_pool = feasible or pool
            return max(source_pool,
                       key=lambda x: self.pool.hamming_distance(x[2], self.run_best_sol))[2]

        if real_T >= self.RANDOM_TARGET_T and pool:
            return random.choice([x[2] for x in pool])

        kick_mode = round_idx % 3
        if kick_mode == 0:
            return self.run_best_sol
        if kick_mode == 1 and pool:
            return pool[0][2]
        return random.choice([x[2] for x in pool]) if pool else self.run_best_sol

    def _build_kick_args(self, strategy, T, round_idx):
        return KickArgs(
            T=T,
            **self.kick_memory.as_kick_args(strategy),
            current_round=round_idx,
            tabu_set=self.pool.kick_tabu_set,
            dyn_bonus=self.base_dyn_bonus,
            global_archive=self.global_archive,
            max_perturb=int(self.ctx.num_pipes * self.MAX_PERTURB_FRACTION),
            **self.rates.as_kick_args(),
        )

    def _note_kick_failure(self, strategy, res, round_idx, use_bandit):
        """Кік нічого не дав: записати це в пам'ять, статистику й лог."""
        self.selector.note_failure(strategy)
        if res.failed_pipe is not None:
            self.kick_memory.note_failure(strategy, res.failed_pipe, round_idx)

        reason = res.message if res.message else "No targets"
        self.ctx.log(f"      -> Kick '{strategy}' failed. Reason: {reason}")

        # TOPO_INV блокує себе сам: коли всі шляхи в табу, стратегія не має
        # що пробувати, доки пам'ять шляхів не очиститься.
        if strategy == "TOPO_INV" and "All paths tabu" in reason:
            self.pool.kick_tabu_set.clear()
            self.ctx.log("      -> [TABU CLEARED] Topo paths memory reset "
                         "to unblock strategy.")

        if use_bandit:
            self.selector.penalise(strategy, StrategySelector.FAILED_KICK_PENALTY)

    def _consensus_locks(self):
        """Труби, однакові в усьому архіві, — їх можна не чіпати глибоким спуском.

        Заморозка вмикається лише наприкінці прогону й лише при холодному
        пошуку: раніше вона зафіксувала б випадковий збіг, а не консенсус.
        Під самий кінець (`progress_ratio > 0.88`) вимикається знову, бо
        останні відсотки виграшу часто лежать саме в «узгоджених» трубах.
        """
        if not (len(self.global_archive) >= self.CONSENSUS_MIN_ARCHIVE
                and self.is_late_game
                and self.temperature_is_cold
                and not self._force_flush_next):
            return set()

        arch_sols = [x[1] for x in self.global_archive]
        raw_consensus = {i for i in range(self.ctx.num_pipes)
                         if all(sol[i] == arch_sols[0][i] for sol in arch_sols)}

        if self.progress_ratio > self.CONSENSUS_OFF_RATIO:
            max_frozen = 0
        else:
            share = (0.25 if self.progress_ratio > 0.6 else 0.10)
            max_frozen = max(5, int(self.ctx.num_pipes * share))

        if max_frozen <= 0:
            return set()
        if len(raw_consensus) > max_frozen:
            # Заморожуємо лише «цікаві» труби — не на межі таблиці діаметрів.
            interesting = [i for i in raw_consensus
                           if 0 < arch_sols[0][i] < self.ctx.max_d_idx]
            return set(list(interesting)[:max_frozen])
        return raw_consensus

    def _polish_kicked(self, strategy, forced_sol, locked, T, base_h_min):
        """Довести збурене рішення до локального оптимуму.

        Спершу дешевий прохід; якщо результат схожий на щось варте уваги —
        глибокий. BASIN_ESCAPE полірувати не можна взагалі: він навмисне
        стрибає в інший басейн, і спуск повернув би його назад.
        """
        if strategy == "BASIN_ESCAPE":
            # Записується ПІДПИС басейну, а не вектор рішення: is_basin_tabu()
            # шукає саме підпис, і для мереж > 50 труб це різні речі.
            if self.run_best_sol:
                self.pool.add_basin_to_tabu(self.run_best_sol)
            return forced_sol

        quick_passes = max(1, 2 - int(T * 2))
        locked_for_squeeze = set() if strategy in self.UNLOCKED_STRATEGIES else locked
        quick_sol = self.ls.gradient_squeeze(
            forced_sol, locked_pipes=locked_for_squeeze, max_passes=quick_passes,
            quick_mode=True, dyn_bonus=self.base_dyn_bonus)
        quick_st = self.ctx.get_cached_stats(quick_sol)
        quick_cost = quick_st.cost

        hb_margin = self.acceptance.heartbeat_margin(self.progress_ratio)
        if not (quick_st.is_feasible(base_h_min)
                and quick_cost < self.run_best_cost * (1.0 + hb_margin)):
            return quick_sol

        gap = (quick_cost - self.run_best_cost) / max(self.run_best_cost, 1.0)
        deep_passes = self.acceptance.deep_passes(gap, T, self.ctx.profile)

        safe_consensus = self._consensus_locks() - (locked if locked else set())
        final_locked = locked_for_squeeze.union(safe_consensus)
        return self.ls.gradient_squeeze(
            quick_sol, locked_pipes=final_locked, max_passes=deep_passes,
            quick_mode=(gap >= 0), dyn_bonus=self.base_dyn_bonus)

    # --- застосування розв'язки -------------------------------------------
    #
    # П'ять методів нижче міняють стан воркера. Порядок мутацій усередині
    # кожного значущий (наприклад, поріг скидання застою рахується від уже
    # оновленого рекорда) і збережений дослівно з часів, коли все це було
    # одним методом на 238 рядків.

    def _accept_record(self, kick, shared_progress):
        diff = self.run_best_cost - kick.cost
        self.run_best_cost, self.run_best_sol = kick.cost, kick.solution
        self.pool.active_pool.insert(0, (kick.score, kick.cost, kick.solution))

        self.stagnation_counter = self._stagnation_after_record(
            diff, self.run_best_cost, self.stagnation_counter)

        if kick.use_bandit:
            improvement_pct = diff / max(self.run_best_cost, 1.0)
            self.selector.reward(kick.strategy,
                                 StrategySelector.record_reward(improvement_pct))
            self.selector.decay_others(kick.strategy,
                                       StrategySelector.OTHERS_DECAY_ON_RECORD)

        if kick.learns_size and not self.ablation:
            self.rates.learn(kick.strategy, kick.used_pct,
                             self.rates.record_alpha(kick.used_pct))

        self.ctx.log(f"   > [FORCE] 💎 Direct Record Update: -${diff:,.0f} "
                     f"({self.run_best_cost/1e6:.4f}M$)")
        self._update_global_best(shared_progress)

    def _accept_escape(self, kick, shared_progress):
        """Велика втеча: пул скидається, рекорд адоптується беззастережно."""
        self.pool.active_pool.clear()
        self.pool.active_pool.append((kick.score, kick.cost, kick.solution))
        self.stagnation_counter = 0
        self.kick_memory.clear()
        self._force_flush_next = True
        self.run_best_cost = kick.cost
        self.run_best_sol = list(kick.solution)
        self.ipc_immunity = 50
        if kick.cost < self.global_best_cost:
            self._update_global_best(shared_progress)
        self.ctx.log(f"      -> POOL FLUSHED. Adopted Major Escape: "
                     f"{kick.cost/1e6:.4f}M$")

        self.selector.reset_after_escape()

        if kick.learns_size and not self.ablation:
            self.rates.learn_escape(kick.used_pct)

    def _accept_into_pool(self, kick, pool_gap, T):
        self.pool.active_pool.append(
            (kick.score * self.POOL_SCORE_PENALTY, kick.cost, kick.solution))

        if kick.use_bandit:
            hamming_dist = self.pool.hamming_distance(kick.solution, self.run_best_sol)
            diversity_ratio = hamming_dist / self.ctx.num_pipes
            if T >= 0.2 and diversity_ratio > 0.02:
                self.selector.reward(
                    kick.strategy,
                    StrategySelector.diversity_reward(diversity_ratio, T))

        if kick.learns_size and not self.ablation:
            self.rates.learn(kick.strategy, kick.used_pct,
                             self.rates.pool_alpha(pool_gap))

        if kick.result.failed_pipe is not None:
            self.kick_memory.note_boost(kick.strategy, kick.result.failed_pipe,
                                        kick.round_idx)

        self.ctx.log(f"      -> Added to Pool (Water Level Accept): "
                     f"{kick.cost/1e6:.4f}M$")

    def _accept_hail_mary(self, kick, base_h_min, shared_progress):
        """Остання спроба: один дешевий прохід може витягнути навіть це."""
        quick = self.ls.gradient_squeeze(kick.solution, max_passes=1, quick_mode=True,
                                         dyn_bonus=self.base_dyn_bonus)
        quick_hail = self.ctx.get_cached_stats(quick)
        q_c = quick_hail.cost

        if quick_hail.is_feasible(base_h_min) and q_c < self.run_best_cost:
            diff = self.run_best_cost - q_c
            self.run_best_cost, self.run_best_sol = q_c, quick
            self.pool.active_pool.insert(0, (q_c, q_c, quick))
            self.stagnation_counter = self._stagnation_after_record(
                diff, self.run_best_cost, self.stagnation_counter)

            if kick.use_bandit:
                self.selector.reward(kick.strategy, StrategySelector.record_reward(
                    diff / max(self.run_best_cost, 1.0)))

            self._update_global_best(shared_progress)
            self.ctx.log(f"     -> 🚀 HAIL MARY → DIRECT RECORD: {q_c/1e6:.4f}M$")
        else:
            self.pool.active_pool.append(
                (kick.score * self.HAIL_MARY_SCORE_PENALTY, kick.cost, kick.solution))
            self.stagnation_counter = int(self.stag_limit * 1.8)
            self.ctx.log(f"     -> 🚀 HAIL MARY ACCEPT: {kick.cost/1e6:.4f}M$")

    def _reject(self, kick):
        self.ctx.log(f"     -> Rejected (Poor or Tabu Basin): {kick.cost/1e6:.4f}M$")
        if kick.learns_size and not self.ablation:
            self.rates.shrink_after_reject(kick.strategy)

    def _apply_kick(self, round_idx, shared_progress):
        """Один раунд диверсифікації: обрати стратегію, вдарити, вирішити долю.

        До етапу 17 це був метод на 238 рядків із CC 79 — найгірший у
        проєкті. Тепер тут лишився сценарій: вибір → удар → лікування →
        поліровка → розв'язка, а кожен крок має ім'я. Порядок обчислень
        збережений дослівно, зокрема те, що `is_ghost_solution` і
        `is_basin_tabu` передаються **функціями**: перша виконує
        симуляції, і обчислювати її завчасно означало б змінити їх
        кількість.
        """
        real_T = self.temperature
        T = self.acceptance.effective_temperature(real_T)

        pool_strats = pool_for(real_T, len(self.global_archive), self._is_tree)
        choice = self.selector.pick(pool_strats, real_T, T, self.is_late_game)
        strategy, use_bandit = choice.strategy, choice.bandit_active

        self.ctx.log(f"{choice.mode_label} (Stag: {self.stagnation_counter}/"
                     f"{self.stag_limit}) -> Applying '{strategy}'...")

        kick_target = self._pick_kick_target(real_T, round_idx)
        spec = KICK_REGISTRY.get(strategy) or KICK_REGISTRY[FALLBACK_KICK]

        try:
            res = getattr(self.kicker, spec.method)(
                kick_target, self._build_kick_args(strategy, T, round_idx))
        except Exception as e:
            self.ctx.log(f"      -> Error in {strategy}: {e}")
            return

        if res.failed:
            self._note_kick_failure(strategy, res, round_idx, use_bandit)
            return

        self.selector.note_success(strategy)

        forced_sol, locked = res.solution, res.locked
        base_h_min = self.ctx.h_min
        explosion_threshold = self.acceptance.explosion_threshold(self.run_best_cost, T)

        _, p, _, _ = self.ctx.get_cached_stats(forced_sol)
        if self.acceptance.is_hopeless_pressure(base_h_min, p, T):
            self.selector.note_failure(strategy)
            return

        if p < self.acceptance.relaxed_h_min(base_h_min, T):
            heal_locks = set() if strategy in self.UNLOCKED_STRATEGIES else locked
            forced_sol, ok, boosts = self.ls.heal_network(forced_sol, heal_locks)
            if not ok:
                if boosts > 0:
                    self.ctx.log("     -> Heal Aborted Early: Cost exceeded limit.")
                return

        if res.message:
            self.ctx.log(f"     -> {res.message}")

        final_sol = self._polish_kicked(strategy, forced_sol, locked, T, base_h_min)

        st = self.ctx.get_cached_stats(final_sol)
        c, p = st.cost, st.p_min
        deficit = max(0.0, base_h_min - p) if st.simulation_ok else float('inf')

        if not (st.simulation_ok and deficit <= self.acceptance.deficit_tolerance(T)):
            return

        effective_cost = self.acceptance.effective_cost(c, deficit, self.run_best_cost)
        if effective_cost > explosion_threshold:
            self.ctx.log(f"      -> Hard-Rejected (Cost Explosion): "
                         f"{effective_cost/1e6:.4f}M$")
            if use_bandit:
                self.selector.penalise(strategy, StrategySelector.EXPLOSION_PENALTY)
            if res.used_pct is not None and not self.ablation:
                self.rates.shrink_after_explosion(strategy)
                self.ctx.log(f"         🧠 [JADE] Shrunk {strategy} size due to explosion.")
            return

        score = (self.ctx.score(effective_cost, p, self.base_dyn_bonus)
                 if deficit == 0 else effective_cost)

        # Рішення приймає `AcceptancePolicy`; застосування лишається тут.
        verdict = self.acceptance.classify(
            is_escape=(strategy == "BASIN_ESCAPE"),
            effective_cost=effective_cost, deficit=deficit,
            best_cost=self.run_best_cost, temperature=T,
            is_basin_tabu=lambda: self.pool.is_basin_tabu(final_sol),
            is_ghost=lambda: self.ctx.is_ghost_solution(final_sol),
        )

        if verdict == Verdict.GHOST_VETO:
            return

        kick = AppliedKick(strategy=strategy, use_bandit=use_bandit, result=res,
                           solution=final_sol, cost=effective_cost, score=score,
                           round_idx=round_idx)

        if verdict == Verdict.RECORD:
            self._accept_record(kick, shared_progress)
        elif verdict == Verdict.ESCAPE:
            self._accept_escape(kick, shared_progress)
        elif verdict == Verdict.POOL:
            self._accept_into_pool(kick, verdict.gap, T)
        elif verdict == Verdict.HAIL_MARY:
            self._accept_hail_mary(kick, base_h_min, shared_progress)
        else:
            self._reject(kick)

        if res.path_signature:
            self.pool.kick_tabu_set.add(res.path_signature)
        self.kick_memory.note_zero_sum(strategy, locked, round_idx)
