"""Оркестрація островів: підняти, дочекатись, зібрати найкраще.

Після етапу 16 у цьому файлі лишився один клас. `SeedFactory` переїхала в
`seeds.py`, `IslandWorker` — в `island.py`; спільного стану між ними не
було, лише спільний файл.
"""

import multiprocessing
import random
import time
from dataclasses import dataclass

from .context import SolverContext
from .island import IslandWorker  # noqa: F401  (сумісність для тестів і experiments/)
from .local_search import LocalSearch
from .pool import SolutionPool
from .seeds import SeedFactory
from .worker import WorkerTask, run_island_worker


@dataclass
class RunState:
    """Стан прогону, який збирають усі етапи `solve_standalone`.

    Три величини жили локальними змінними в методі на 221 рядок і саме
    тому його не можна було розділити: кожен шматок читав і писав їх.
    Окремий тип робить цю залежність видимою — і дозволяє передати стан
    у винесений метод, не повертаючи кортеж із трьох елементів.
    """

    best_cost: float = float('inf')
    best_sol: list | None = None
    #: Симуляції островів: лічильник головного процесу їх не бачить, бо
    #: кожен воркер має власний контекст.
    worker_sims: int = 0


class AnalyticalSolver:
    def __init__(self, simulator_instance, available_diameters, v_opt=1.0, max_sims=None, time_limit_sec=None, pool=None, log_dir=None, n_workers=1):
        self.ctx = SolverContext(simulator_instance, available_diameters, v_opt)
        self.ls = LocalSearch(self.ctx)
        self.mp_pool = pool
        self.log_dir = log_dir
        self.n_workers = n_workers

        # `network_class` і `BEAM_WIDTH` прибрані на етапі 19: перше не
        # читалось узагалі, друге існувало лише щоб передати його в
        # `SeedFactory`, яка його ігнорувала.
        self.seeder = SeedFactory(self.ctx, self.ls)
        self.pool = SolutionPool(self.ctx)

        self.BASE_SIM_BUDGET = self.ctx.profile.sim_budget

        if max_sims is not None:
            self.max_sims = max_sims
        else:
            self.max_sims = self.BASE_SIM_BUDGET

        if time_limit_sec is not None:
            self.time_limit_sec = time_limit_sec
        else:
            cluster_speed = max(self.ctx.sim_speed, 1.0) * self.n_workers
            self.time_limit_sec = (self.max_sims / cluster_speed) * 5

    def solve_fast(self, ui_callback=None):
        print("\n[AnalyticalSolver] ⚡ Initiating FAST Analytical Search (Velocity Sweep)...\n")
        start_time = time.time()

        print("   > Sweeping ideal velocities (0.5 to 2.0 m/s)...")
        seeds = self.seeder.make_diverse_seeds_for_fast()

        print("   > Sweeping Backbone (Trunk-Branch) configurations...")
        for v in [0.8, 1.0, 1.2, 1.5, 1.8]:
            bb_seed = self.seeder.make_backbone_seed(target_v=v)
            if bb_seed:
                seeds.append(bb_seed)

        valid_seeds = []
        seen_sigs = set()

        for sol in seeds:
            c, p, _, _ = self.ctx.get_cached_stats(sol)
            if p >= self.ctx.h_min:
                sig = tuple(sol)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    valid_seeds.append((c, list(sol)))

        global_best_cost = float('inf')
        global_best_sol = None

        if valid_seeds:
            valid_seeds.sort(key=lambda x: x[0])
            top_n = min(5, len(valid_seeds))

            print(f"   > Sweep finished. Found {len(valid_seeds)} valid seeds. Deep Polishing TOP-{top_n}...")

            for rank in range(top_n):
                raw_cost, raw_sol = valid_seeds[rank]

                polished = self.ls.gradient_squeeze(raw_sol, max_passes=None, quick_mode=False, dyn_bonus=raw_cost * 0.001)
                p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)

                if p_p >= self.ctx.h_min and p_cost < global_best_cost:
                    global_best_cost = p_cost
                    global_best_sol = polished
                    print(f"     [{rank+1}/{top_n}] Polished {raw_cost/1e6:.4f}M$ ➡️ RECORD: {global_best_cost/1e6:.4f}M$")
                else:
                    print(f"     [{rank+1}/{top_n}] Polished {raw_cost/1e6:.4f}M$ ➡️ {p_cost/1e6:.4f}M$ (Discarded)")

                if ui_callback is not None:
                    ui_callback(self.ctx.sim_count, global_best_cost, global_best_sol)

        if global_best_sol is None:
            print("\n[WARNING] Fast seeds infeasible. Falling back to max diameters.")
            global_best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            global_best_cost, _, _, _ = self.ctx.get_cached_stats(global_best_sol)

        total_time = time.time() - start_time
        print(f"\n[Fast Analytical] FINAL RESULT: {global_best_cost/1e6:.4f}M$ (Time: {total_time:.2f}s | Sims: {self.ctx.sim_count})")

        self.history = [(self.ctx.sim_count, global_best_cost)]
        if ui_callback is not None:
            ui_callback(self.ctx.sim_count, global_best_cost, global_best_sol)

        real_diams = [self.ctx.diameters[i] for i in global_best_sol]
        return real_diams

    def _worker_quota(self):
        """Скільки симуляцій дістається одному острову."""
        if self.max_sims == float('inf'):
            return float('inf')
        return int(self.max_sims // self.n_workers)

    def _build_tasks(self, run_time_budget, worker_sim_quota, shared_progress,
                     global_archive):
        """Завдання для островів.

        `seed_modifier` береться з RNG головного процесу — саме через нього
        `--seed` доходить до воркерів (§12.4). Тому він рахується тут, у
        тому самому місці послідовності викликів, що й раніше: зсув цього
        виклику змінює всі зерна воркерів і робить прогін іншим.
        """
        seed_modifier = random.randint(1, 10000)
        return [
            WorkerTask(
                diameters=self.ctx.diameters, v_opt=self.ctx.v_opt,
                time_budget=run_time_budget, global_best_cost=float('inf'),
                seed=seed_modifier + i, worker_id=i, n_workers=self.n_workers,
                max_sims=worker_sim_quota, shared_progress=shared_progress,
                log_dir=self.log_dir, global_archive=global_archive,
            )
            for i in range(self.n_workers)
        ]

    @staticmethod
    def _dedup_archive(live_archive, limit=6):
        """Найкращі різні рішення островів — те, з чого воркери беруть донорів.

        Чиста функція: тестується без пулу, симулятора й процесів.
        """
        live_archive.sort(key=lambda x: x[0])
        unique_archive = []
        seen_sigs = set()
        for cost, sol in live_archive:
            sig = tuple(sol)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                unique_archive.append((cost, sol))
                if len(unique_archive) >= limit:
                    break
        return unique_archive

    def _live_snapshot(self, shared_progress, best_cost, best_sol):
        """Що острови повідомили про себе на цю мить.

        Читає `shared_progress` і нічого не змінює — тому тестується з
        простим словником, без жодної симуляції. Раніше ці сорок рядків
        жили всередині циклу моніторингу, і перевірити їх можна було лише
        реальним паралельним прогоном.

        Повертає `(live_best, live_best_sol, total_sims, status_parts,
        live_archive)`; порядок читання джерел збережено дослівно, бо від
        нього залежить, чиє рішення переможе за рівної вартості.
        """
        live_best, live_best_sol = best_cost, best_sol
        live_archive = []
        status_parts = []
        total_sims = 0

        gb = shared_progress.get('global_best')
        if gb:
            live_archive.append(gb)
            if gb[0] < live_best:
                live_best = gb[0]
                live_best_sol = gb[1]

        for wid in range(self.n_workers):
            prog = shared_progress.get(wid, 0)
            if isinstance(prog, dict):
                sims = prog.get('sims', 0)
                w_best = prog.get('best_cost', float('inf'))

                if w_best < live_best:
                    live_best = w_best
                    if 'best_sol' in prog:
                        live_best_sol = prog['best_sol']
            else:
                sims = prog

            total_sims += sims
            if sims > 0:
                status_parts.append(f"W{wid+1}:{sims//1000}k")
            else:
                status_parts.append(f"W{wid+1}:--")

            w_sol = shared_progress.get(f'best_sol_{wid}')
            if w_sol:
                live_archive.append(w_sol)

        return live_best, live_best_sol, total_sims, status_parts, live_archive

    def _monitor_parallel(self, async_results, shared_progress, state,
                          start_time, ui_callback):
        """Цикл живого моніторингу: читає прогрес островів, поки вони працюють.

        Пошуку не веде — читає `shared_progress`, друкує рядок стану,
        оновлює `self.history` і віддає прогрес у GUI. Виносити його було
        безпечно саме тому: єдине, що він мутує поза собою, — це
        `state` і `self.history`.
        """
        last_print_time = 0
        print_interval = self.ctx.profile.print_interval
        last_sent_cost = float('inf')

        while True:
            if all(res.ready() for _, res in async_results):
                break

            curr_time = time.time()
            if curr_time - last_print_time >= print_interval:
                last_print_time = curr_time
                m, s = divmod(int(curr_time - start_time), 60)

                (live_best, live_best_sol, total_sims,
                 status_parts, live_archive) = self._live_snapshot(
                    shared_progress, state.best_cost, state.best_sol)

                best_str = (f"{live_best/1e6:.4f}M$" if live_best != float('inf')
                            else "---")
                print(f"   > [Live {m:02d}:{s:02d}] Best: {best_str} | "
                      f"Sims: {total_sims/1000:.1f}k | {' '.join(status_parts)}")

                current_total = self.ctx.sim_count + state.worker_sims + total_sims

                if live_best < state.best_cost:
                    state.best_cost = live_best
                    state.best_sol = live_best_sol

                if live_best != float('inf'):
                    self.history.append((current_total, live_best))

                if ui_callback is not None:
                    if state.best_cost < last_sent_cost and state.best_sol is not None:
                        last_sent_cost = state.best_cost
                        ui_callback(current_total, state.best_cost, list(state.best_sol))
                    else:
                        ui_callback(current_total, state.best_cost, None)

                if live_archive:
                    shared_progress['global_archive'] = self._dedup_archive(live_archive)

            time.sleep(1.0)

    def _run_islands(self, tasks, shared_progress, state, start_time, ui_callback):
        """Провести острови й зібрати їхні рішення.

        Дві гілки — паралельна й послідовна — відрізняються лише способом
        запуску; обидві повертають однаковий список `(вартість, рішення)`.
        """
        island_results = []

        if self.mp_pool:
            async_results = [
                (t.worker_id, self.mp_pool.apply_async(run_island_worker, (t,)))
                for t in tasks
            ]
            self._monitor_parallel(async_results, shared_progress, state,
                                   start_time, ui_callback)

            for wid, res in async_results:
                try:
                    result = res.get()
                    if result.best_solution is not None:
                        island_results.append((result.best_cost, result.best_solution))
                except Exception as e:
                    print(f"     [Error] Worker {wid+1} crashed: {e}")

            if shared_progress is not None:
                for wid in range(self.n_workers):
                    prog = shared_progress.get(wid, {})
                    if isinstance(prog, dict):
                        state.worker_sims += prog.get('sims', 0)
        else:
            for t in tasks:
                try:
                    result = run_island_worker(t)
                    if result.best_solution is not None:
                        island_results.append((result.best_cost, result.best_solution))
                    state.worker_sims += result.sim_count
                    print(f"   > Worker {t.worker_id+1} Finished. "
                          f"Best: {result.best_cost/1e6:.4f}M$")
                except Exception as e:
                    print(f"     [Error] Sequential Worker {t.worker_id+1} crashed: {e}")

        return island_results

    def _adopt_best(self, state, island_results, shared_progress):
        """Взяти найкраще з трьох джерел — у тому самому порядку, що й раніше.

        Порядок значущий: за рівної вартості перемагає джерело, прочитане
        пізніше. `island_results` — те, що воркери повернули напряму; у
        послідовному режимі це **єдине** джерело, і без нього рішення
        воркера втрачалось (§2.3.2).
        """
        for c, sol in island_results:
            if sol is not None and c < state.best_cost:
                state.best_cost = c
                state.best_sol = list(sol)

        if shared_progress is None:
            return

        gb = shared_progress.get('global_best')
        if gb and gb[0] < state.best_cost:
            state.best_cost = gb[0]
            state.best_sol = list(gb[1])

        for wid in range(self.n_workers):
            w_sol = shared_progress.get(f'best_sol_{wid}')
            if w_sol and w_sol[0] < state.best_cost:
                state.best_cost = w_sol[0]
                state.best_sol = list(w_sol[1])

    def _final_polish(self, state):
        """Останній глибокий спуск по чемпіону — або запасне рішення."""
        print("\n[FINAL POLISH] Polishing global best solution...")
        if state.best_sol:
            polished = self.ls.gradient_squeeze(
                state.best_sol, max_passes=None, quick_mode=False,
                dyn_bonus=state.best_cost * 0.001)
            p_cost, p_p, _, _ = self.ctx.get_cached_stats(polished)
            if p_p >= self.ctx.h_min and p_cost < state.best_cost:
                state.best_cost = p_cost
                state.best_sol = polished
                print(f"   > [POLISH] Improved! Final: {state.best_cost/1e6:.4f}M$")

            self.history.append((self.ctx.sim_count + state.worker_sims,
                                 state.best_cost))
        else:
            print("\n[WARNING] No valid solution found. Returning safe default.")
            state.best_sol = [self.ctx.max_d_idx] * self.ctx.num_pipes
            try:
                state.best_cost, _, _, _ = self.ctx.get_cached_stats(state.best_sol)
            except Exception as e:
                print(f"[WARNING] Не вдалось оцінити запасне рішення: {e}")
                state.best_cost = float('inf')

    def solve_standalone(self, ui_callback=None):
        """Острівна модель: підняти острови, зібрати найкраще, відполірувати.

        До етапу 15 це був один метод на 221 рядок із CC 59 — сім справ
        поспіль, із яких найбільша (цикл живого моніторингу) не приймала
        жодного рішення про пошук. Тепер оркестрація тут, а кожна справа
        має ім'я; порядок обчислень збережено дослівно, тому еталон
        збігається побітово.
        """
        print("\n[AnalyticalSolver] ⚡ Initiating Continuous Island Model Search...\n")
        start_time = time.time()
        state = RunState()
        global_archive = []
        self.history = []

        # Епох більше немає: острівна модель із асинхронною синхронізацією
        # замінила їх на один прогін із міграцією (§2.3).
        run_time_budget = self.time_limit_sec

        worker_sim_quota = self._worker_quota()
        quota_str = "∞" if worker_sim_quota == float('inf') else f"{worker_sim_quota:,}"
        print(f"  [Quota] Allocated {quota_str} sims per worker.\n")

        manager = multiprocessing.Manager() if self.mp_pool else None
        shared_progress = manager.dict() if manager else None
        if shared_progress:
            for i in range(self.n_workers):
                shared_progress[i] = 0
            shared_progress['global_archive'] = []

        island_results = []
        try:
            mode_str = "PARALLEL" if self.mp_pool else "SEQUENTIAL"
            print("="*46)
            print(f" [CONTINUOUS RUN] {mode_str} Workers: {self.n_workers} | "
                  f"Time Limit: {run_time_budget/60:.1f} min")
            print("="*46)

            tasks = self._build_tasks(run_time_budget, worker_sim_quota,
                                      shared_progress, global_archive)
            island_results = self._run_islands(tasks, shared_progress, state,
                                               start_time, ui_callback)
        except KeyboardInterrupt:
            print("\n\n[AnalyticalSolver] 🛑 Отримано сигнал переривання (Ctrl+C)!")
            print("[AnalyticalSolver] М'яка зупинка. Перехід до генерації звітів...")

        self._adopt_best(state, island_results, shared_progress)
        self._final_polish(state)

        total_time = time.time() - start_time
        total_cluster_sims = self.ctx.sim_count + state.worker_sims

        if state.best_cost != float('inf'):
            print(f"\n[AnalyticalSolver] FINAL RESULT: {state.best_cost/1e6:.4f}M$ "
                  f"(Total Time: {total_time/60:.1f}m | "
                  f"Total Sims: {total_cluster_sims:,})")
        else:
            print("\n[AnalyticalSolver] EXECUTION ABORTED. No valid solutions.")

        if not state.best_sol:
            return []
        return [self.ctx.diameters[max(0, min(i, self.ctx.max_d_idx))]
                for i in state.best_sol]
