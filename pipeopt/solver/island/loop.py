"""Острів: складання домішок і головний цикл.

`IslandWorker` зібраний із чотирьох домішок так само, як `KickStrategies`
із трьох (етап 16): диспетчер кіків викликає стратегії **за іменем**
через `getattr`, тому делегування довелося б або дублювати у фасаді, або
ускладнювати реєстр. Домішки лишають виклики тими самими, а файли —
окремими.

До розділення тут був один клас на 1210 рядків із 34 методами; шукати в
ньому потрібний метод можна було лише пошуком по тексту.
"""

import random
import time

import networkx as nx

from ..acceptance import AcceptancePolicy
from ..adaptive_rates import AdaptiveRates
from ..kick_memory import KickMemory
from ..kick_registry import (
    TRACKED_STRATEGIES,
)
from ..pool import SolutionPool
from ..seeds import SeedFactory
from ..strategy_selector import StrategySelector

from .beam import BeamSearch
from .diversification import Diversification
from .exchange import IslandExchange
from .state import AppliedKick, IslandState

__all__ = ["AppliedKick", "IslandWorker"]


class IslandWorker(IslandState, IslandExchange,
                   Diversification, BeamSearch):
    """Один острів острівної моделі: веде промінь і б'є кіками."""
    def __init__(self, ctx, kicker, local_search, worker_id, n_workers,
                 max_sims, beam_width, global_archive, ablation=False):
        # `network_class` прибрано на етапі 19: поле зберігалось і не
        # читалось. Клас мережі доступний через `ctx.profile.name`.
        self.ctx = ctx
        self.kicker = kicker
        self.ls = local_search
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.max_sims = max_sims
        self.BEAM_WIDTH = beam_width
        self.global_archive = global_archive

        self.pool = SolutionPool(self.ctx)
        self.seeder = SeedFactory(self.ctx, self.ls)

        # Деревоподібність мережі не змінюється, тому обчислюється один раз,
        # а не на кожному застосуванні кіка.
        self._is_tree = nx.is_tree(self.ctx.base_G_flow)

        # Раніше цей перелік існував у трьох місцях (self.strategies,
        # all_tracked і три пули в _apply_kick) і містив стратегію
        # "SUBMARINE", якої немає в диспетчері — тобто їй марно велася
        # статистика UCB1. Тепер джерело одне — реєстр.
        # Ablation-режим: фіксує T, знімає послаблення вимоги тиску й
        # вимикає самонавчання. Раніше це була локальна константа в тілі
        # `_apply_kick`, тобто шість недосяжних гілок. Тепер це параметр,
        # і ablation-прогін не вимагає правки вихідника.
        self.ablation = ablation
        self.selector = StrategySelector(TRACKED_STRATEGIES, ablation=ablation)
        self.acceptance = AcceptancePolicy(ablation=ablation)

        #: Ширина раунду: таблиця за класом мережі лишається **стартовим**
        #: значенням, а робочу ширину далі визначає заміряна ціна раунду
        #: (`narrow_round`).
        self.SINGLE_CANDIDATES = self.ctx.profile.single_candidates

        #: Межа ціни раунду рахується в `run()`, коли відомий бюджет часу.
        #: До того правило інертне: нескінченна межа означає «не звужувати».
        self._round_cost_limit = float('inf')

        self.stagnation_counter = 0
        self.run_best_cost = float('inf')
        self.run_best_sol = None
        self.global_best_cost = float('inf')
        self.last_published_cost = float('inf')
        self.progress_ratio = 0.0
        self.is_late_game = False

        self.last_injected_peer = {}
        self.rescue_fired = False
        self._force_flush_next = False
        self.ipc_immunity = 30
        # Чотири короткі пам'яті кіків, які створювались, очищались і
        # передавались у кік разом — тепер один об'єкт.
        self.kick_memory = KickMemory()

        # Розміри кроку кіків і правила їх навчання (JADE) — раніше
        # чотири поля, які оновлювались у п'яти місцях `_apply_kick`,
        # кожне через власну копію ланцюга if/elif по стратегії.
        self.rates = AdaptiveRates()

        # Ці поля читалися дев'ятьма методами (`base_dyn_bonus`), п'ятьма
        # (`stag_limit`) і одним (`reserve_pool`), але присвоювалися лише в
        # `run()` — тобто до її виклику об'єкт був у невалідному стані, і
        # будь-яке звернення до методу давало AttributeError у глибині кіка.
        # `run()` перераховує їх перед першим кіком і далі щороунду, тому ці
        # значення ніколи не доходять до пошуку: вони існують, щоб об'єкт був
        # валідним з моменту створення (і щоб методи можна було тестувати).
        self.base_dyn_bonus = 0.0
        self.stag_limit = 4
        self.reserve_pool = []
        self._last_ping_sims = 0
        self._last_ping_time = 0.0
        self._migration_interval_sims = 15000
        self._last_migration_sims = 0

    def run(self, time_budget, global_best_cost, shared_progress):
        self.global_best_cost = global_best_cost
        start_time = time.time()
        self.pool.clear_all()

        # Раніше тут була гілка під `epoch != 0`, яка звала `make_warm_seeds`
        # (96 рядків, CC 37). Епох більше немає — гілка була недосяжна.
        seeds = self.seeder.make_diverse_seeds()
        self.reserve_pool = self.seeder.make_reserve_pool(size=4)

        best_initial_cost = min([self.ctx.get_cached_stats(s)[0] for s in seeds]) if seeds else global_best_cost
        self.base_dyn_bonus = min(best_initial_cost, global_best_cost) * 0.001

        self._initialize_seeds(seeds)

        self.ipc_immunity = 30
        self.stag_limit = 4
        round_idx = 0
        run_start_sims = self.ctx.sim_count
        just_flushed = False

        self._last_ping_sims = self.ctx.sim_count
        self._last_ping_time = time.time()

        # Блок, що стояв тут до цього, не мав ефекту: `progress_ratio` на
        # цей момент завжди 0.0, тому гілка `> 0.8` була недосяжна, а при
        # скінченному бюджеті значення все одно перераховувалось у циклі
        # до першої перевірки інтервалу. Розрахунок тепер один — у
        # `_migration_interval()`.
        self._migration_interval_sims = self._migration_interval()
        self._last_migration_sims = self.ctx.sim_count

        # Ширина раунду підлаштовується під **заміряну** його ціну
        # (§`IslandState.ROUND_BUDGET_SHARE`). Межа рахується один раз:
        # бюджет прогону за час прогону не змінюється.
        self._round_cost_limit = self.round_cost_limit(time_budget)

        def update_progress_ping():
            if shared_progress is not None:
                curr_sims = self.ctx.sim_count
                curr_time = time.time()

                if (curr_sims - self._last_ping_sims) > 10000 or (curr_time - self._last_ping_time) > 2.0:
                    self._last_ping_sims = curr_sims
                    self._last_ping_time = curr_time

                    current_data = self._read_shared(shared_progress,
                                                     self.worker_id)
                    if isinstance(current_data, dict):
                        new_data = dict(current_data)
                        new_data['sims'] = curr_sims
                        self._publish(shared_progress, self.worker_id, new_data)

        self.ls.progress_callback = update_progress_ping

        while True:
            self.pool.current_round = round_idx
            self.base_dyn_bonus = min(self.run_best_cost, self.global_best_cost) * 0.001 * random.uniform(0.95, 1.05)

            gb = shared_progress.get('global_best') if shared_progress else None
            if shared_progress is not None:
                self._process_ipc(shared_progress, gb)
                self._check_rescue(gb)
                self._publish(shared_progress, self.worker_id,
                              {"round": round_idx + 1,
                               "sims": self.ctx.sim_count,
                               "best_cost": self.run_best_cost})

            elapsed = time.time() - start_time
            sims_used = self.ctx.sim_count - run_start_sims
            if elapsed > time_budget or sims_used >= self.max_sims: break

            just_flushed = self.rescue_fired
            self.rescue_fired = False

            round_start_sims = self.ctx.sim_count

            self.progress_ratio = self.progress_of(sims_used, elapsed, time_budget)
            self.is_late_game = self.progress_ratio > 0.5

            self.stag_limit = max(4, int(12 * (1.0 - self.progress_ratio)))

            self._migration_interval_sims = self._migration_interval()

            self._check_mini_restart(gb)
            if round_idx > 0 and round_idx % 6 == 0: self.pool.kick_tabu_set.clear()

            self._apply_swap(round_idx, shared_progress)

            if self.stagnation_counter >= 2:
                self._apply_kick(round_idx, shared_progress)

            if self._force_flush_next:
                just_flushed = True
                self._force_flush_next = False

            next_gen = self._generate_mutations()

            if next_gen:
                next_gen.sort(key=lambda x: x[0])
                just_flushed = self._beam_search_and_update(next_gen, just_flushed, shared_progress, round_idx)
            else:
                self._emergency_respawn()
                just_flushed = True

            if self.narrow_round(self.ctx.sim_count - round_start_sims):
                self.ctx.log(
                    f"   [ROUND] Раунд з'їв понад {self.ROUND_BUDGET_SHARE:.0%} "
                    f"бюджету; кандидатів на батька -> {self.SINGLE_CANDIDATES}")

            self._maybe_migrate(gb, shared_progress)
            round_idx += 1

        current_data = self._read_shared(shared_progress, self.worker_id, {})
        if isinstance(current_data, dict):
            current_data['sims'] = self.ctx.sim_count
            current_data['best_cost'] = self.run_best_cost
            self._publish(shared_progress, self.worker_id, current_data)

        self.ls.progress_callback = None

        # Тіснота обмеження тиску на цій мережі: скільки ходів промахнулось
        # повз обмеження і наскільки глибоко. Величина нічого не вирішує —
        # вона існує, щоб було видно, з якою мережею має справу солвер
        # (§24 REFACTORING_PLAN.md). Замір безкоштовний: недобір уже
        # обчислений у `evaluate_candidate`.
        tightness = self.ls.tightness
        if tightness.infeasible:
            bands = ", ".join(f"{k} {v:.0%}" for k, v in tightness.histogram().items())
            self.ctx.log(
                f"   [TIGHTNESS] Недопустимих ходів: {tightness.infeasible_share:.1%} "
                f"({tightness.infeasible} з {tightness.evaluated}); "
                f"найменший недобір {tightness.smallest_deficit:.3f} м; {bands}")
        elif tightness.evaluated:
            self.ctx.log(
                f"   [TIGHTNESS] Жоден із {tightness.evaluated} ходів не порушив "
                "тиску — обмеження на цій мережі не тісне.")

        _, final_p, _, _ = self.ctx.get_cached_stats(self.run_best_sol)
        if final_p < self.ctx.h_min:
            self.ctx.log(" ⚠️ [FINAL GUARD] Final solution strictly invalid. Applying hard heal.")
            healed_sol, ok, _ = self.ls.heal_network(self.run_best_sol, set())
            if ok:
                self.run_best_sol = healed_sol
                self.run_best_cost, _, _, _ = self.ctx.get_cached_stats(healed_sol)
            else:
                self.run_best_cost = float('inf')

        return self.run_best_cost, self.run_best_sol
