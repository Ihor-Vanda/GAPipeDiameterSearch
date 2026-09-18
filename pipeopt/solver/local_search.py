import itertools
import random
import math


class DescentStats:
    """Врожай проходів градієнтного спуску.

    Перша величина, яку солвер міряє **сам про себе** — і яка потрібна,
    щоб позбутись порогів «за класом мережі» (§24.4). Замір §22.2
    показав: у зійшовшомуся стані поліпшуваних труб нуль із 454, а прохід
    по них коштує Θ(n) симуляцій. Врожай (`accepted / scanned`) — це той
    самий факт у вигляді числа, доступного під час прогону: коли він
    падає до нуля, повні проходи перестають себе виправдовувати.

    Поки що лише лічильники: жодне рішення від них не залежить, і
    поведінка пошуку не змінюється. Правило, яке їх читає, — наступний
    крок (§23.2), і воно вимагає окремого A/B-заміру.
    """

    __slots__ = ("passes", "scanned", "skipped", "accepted", "last_pass_yield")

    def __init__(self):
        self.passes = 0
        #: Труби, для яких хід справді перевірявся симуляцією.
        self.scanned = 0
        #: Труби, пропущені евристикою `quick_mode` без перевірки.
        self.skipped = 0
        self.accepted = 0
        #: Врожай останнього завершеного проходу — саме він цікавий
        #: правилу зупинки: середнє за весь прогін згладжує момент, коли
        #: спуск уже зійшовся.
        self.last_pass_yield = None

    @property
    def yield_rate(self):
        """Частка перевірених труб, для яких хід було прийнято."""
        if not self.scanned:
            return None
        return self.accepted / self.scanned

    def start_pass(self):
        self.passes += 1
        return self.scanned, self.accepted

    def finish_pass(self, marks):
        scanned_before, accepted_before = marks
        scanned = self.scanned - scanned_before
        if scanned:
            self.last_pass_yield = (self.accepted - accepted_before) / scanned

    def note_scan(self):
        self.scanned += 1

    def note_skip(self):
        self.skipped += 1

    def note_accept(self):
        self.accepted += 1

    def snapshot(self):
        return {"passes": self.passes, "scanned": self.scanned,
                "skipped": self.skipped, "accepted": self.accepted,
                "yield_rate": self.yield_rate,
                "last_pass_yield": self.last_pass_yield}


class ConstraintTightness:
    """Наскільки тісне обмеження тиску на цій конкретній мережі.

    Замінює механізм, який не спрацював. Вікно відносної допустимості
    (readme.md §2 до 15.09) виходило з припущення, що кандидати часто
    промахуються повз обмеження «трохи». Замір на трьох мережах показав,
    що «трохи» — величина, яка відрізняється в рази, а на деяких мережах
    недопустимих кандидатів немає взагалі:

    | мережа | недопустимих | медіана недобору |
    |---|---|---|
    | Ханої (34 труби, 4 цикли) | 23.7 % | 0.844 м |
    | Балерма (454 труби, 12 циклів) | 18.7 % | 0.112 м |
    | Фосс (58 труб, 23 цикли) | 0 % | — |

    Тобто масштаб задає мережа, а не константа в коді. Ці лічильники
    роблять його спостережуваним: недобір **уже обчислений** у
    `evaluate_candidate`, тому замір не коштує жодної симуляції.

    Навіщо це солверу, а не лише звіту: §24 `REFACTORING_PLAN.md` вимагає
    величин, які алгоритм міряє сам про себе, щоб позбутись порогів «за
    класом мережі». Тіснота обмеження — одна з них: на мережі, де
    недопустимих кандидатів нуль, спуску нема сенсу берегтися тиску, а
    там, де їх чверть, — навпаки.

    Поведінки не змінює: лише рахує.
    """

    #: Межі гістограми недобору в метрах. Обрані так, щоб покрити
    #: спостережений діапазон: на Балермі все лежить під 0.5, на Ханої
    #: половина — за 1.0.
    BANDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

    __slots__ = ("evaluated", "infeasible", "smallest_deficit", "_bands")

    def __init__(self):
        self.evaluated = 0
        self.infeasible = 0
        #: Найменший недобір (найближчий до нуля) — саме він показує, чи
        #: існує взагалі «легке» порушення на цій мережі.
        self.smallest_deficit = None
        self._bands = [0] * (len(self.BANDS) + 1)

    def note(self, surplus):
        self.evaluated += 1
        if surplus >= 0.0:
            return
        deficit = -surplus
        self.infeasible += 1
        if self.smallest_deficit is None or deficit < self.smallest_deficit:
            self.smallest_deficit = deficit
        for i, edge in enumerate(self.BANDS):
            if deficit < edge:
                self._bands[i] += 1
                return
        self._bands[-1] += 1

    @property
    def infeasible_share(self):
        if not self.evaluated:
            return None
        return self.infeasible / self.evaluated

    def histogram(self):
        """Частка недопустимих кандидатів за глибиною недобору."""
        if not self.infeasible:
            return {}
        labels = [f"<{e:g}" for e in self.BANDS] + [f">={self.BANDS[-1]:g}"]
        return {k: v / self.infeasible for k, v in zip(labels, self._bands) if v}

    def snapshot(self):
        return {"evaluated": self.evaluated, "infeasible": self.infeasible,
                "infeasible_share": self.infeasible_share,
                "smallest_deficit": self.smallest_deficit,
                "histogram": self.histogram()}


class LocalSearch:
    #: Скільки проходів дає внутрішня поліровка кандидата в
    #: `evaluate_candidate`. Її робота — **розставити кандидатів по
    #: місцях**, а не довести кожного до кінця: глибоко полірує
    #: промінь, і лише тих, хто в нього потрапив (`island/beam.py`:
    #: п'ять проходів лідеру, три — наступним двом). Довші проходи
    #: тут майже не міняють порядку, але коштують бюджету: за тих
    #: самих 150 тис. симуляцій один прохід дає в 1.2 раза більше
    #: кандидатів на Ханої і в 2.0-2.5 раза — на Балермі
    #: (журнал змін, 2026-09-16, етап 25).
    RANKING_POLISH_PASSES = 1

    def __init__(self, ctx):
        self.ctx = ctx
        #: Накопичується за весь прогін; поведінки не змінює (§25.6).
        self.stats = DescentStats()
        #: Тіснота обмеження тиску на цій мережі — теж лише замір.
        self.tightness = ConstraintTightness()

    def get_high_impact_pipes(self, indices, top_k):
        unit_losses = self.ctx.get_cached_heuristics(indices)
        impact_scores = []
        for i in range(self.ctx.num_pipes):
            d_idx = indices[i]
            if d_idx > 0:
                save_potential = self.ctx.lengths[i] * (self.ctx.costs_array[d_idx] - self.ctx.costs_array[d_idx-1])
            else:
                save_potential = 0

            risk = max(unit_losses[i], 1e-5)
            impact = save_potential * (1.0 + risk)
            impact_scores.append((i, impact))

        impact_scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in impact_scores[:top_k]]

    def gradient_squeeze(self, indices, locked_pipes=None, max_passes=None, quick_mode=False,
                         dyn_bonus=None, min_rel_improvement=0.0003, classical_descent=False):
        """Градієнтний спуск по індексах діаметрів.

        `classical_descent=True` — базовий режим для ablation-порівнянь:
        класичний жадібний спуск без гідравлічних евристик, без обмеження
        кількості проходів, без дострокового переривання та без блокувань.
        (Раніше цей режим вмикався правкою вихідника.)
        """
        if dyn_bonus is None:
            c_start, _, _, _ = self.ctx.get_cached_stats(indices)
            dyn_bonus = c_start * 0.001

        locked_pipes = locked_pipes or set()

        if classical_descent:
            quick_mode = False
            max_passes = None
            min_rel_improvement = 0.0
            locked_pipes = set()

        current_indices = list(indices)
        improved = True
        passes = 0

        cost, p_min, _, _ = self.ctx.get_cached_stats(current_indices)
        best_score = self.ctx.score(cost, p_min, dyn_bonus)

        milestone_cost = cost
        milestone_pass = 0

        active_indices = [i for i in range(self.ctx.num_pipes) if i not in locked_pipes]

        while improved:
            improved = False
            passes += 1
            pass_marks = self.stats.start_pass()

            if getattr(self, 'progress_callback', None):
                self.progress_callback()

            if max_passes and passes > max_passes: break

            if passes - milestone_pass >= 3:
                rel_improvement = (milestone_cost - cost) / max(milestone_cost, 1.0)
                if rel_improvement < min_rel_improvement:
                    break
                milestone_cost = cost
                milestone_pass = passes

            random.shuffle(active_indices)
            active_indices_pass = active_indices

            if quick_mode:
                unit_losses = self.ctx.get_cached_heuristics(current_indices)
            else:
                unit_losses = None

            p_surplus = p_min - self.ctx.h_min
            is_critically_tight = p_surplus < 0.1

            for idx in active_indices_pass:
                curr_d = current_indices[idx]

                if quick_mode:
                    loss_threshold = 0.02 if is_critically_tight else 0.10
                    crit_threshold = 0.005 if is_critically_tight else 0.02

                    if unit_losses[idx] >= loss_threshold:
                        self.stats.note_skip(); continue
                    if is_critically_tight and unit_losses[idx] >= crit_threshold:
                        self.stats.note_skip(); continue

                self.stats.note_scan()

                best_local_sol = None
                best_local_score = best_score
                best_local_cost = cost
                best_local_p = p_min

                if curr_d > 0:
                    test_sol = list(current_indices)
                    test_sol[idx] -= 1
                    c, p_val, _, _ = self.ctx.get_cached_stats(test_sol)
                    if p_val >= self.ctx.h_min:
                        score = self.ctx.score(c, p_val, dyn_bonus)
                        if score < best_local_score or (quick_mode and c < best_local_cost):
                            best_local_score, best_local_sol, best_local_cost, best_local_p = score, test_sol, c, p_val

                if curr_d < self.ctx.max_d_idx and not quick_mode:
                    test_sol = list(current_indices)
                    test_sol[idx] += 1
                    c, p_val, _, _ = self.ctx.get_cached_stats(test_sol)
                    if p_val >= self.ctx.h_min:
                        score = self.ctx.score(c, p_val, dyn_bonus)
                        if score < best_local_score:
                            best_local_score, best_local_sol, best_local_cost, best_local_p = score, test_sol, c, p_val

                if best_local_sol is not None:
                    current_indices = best_local_sol
                    best_score, cost, p_min = best_local_score, best_local_cost, best_local_p
                    improved = True
                    self.stats.note_accept()

            self.stats.finish_pass(pass_marks)

        return current_indices

    def heal_network(self, kicked, locked_pipes):
        locked_pipes_local = set(locked_pipes)
        boosts = 0

        while True:
            _, min_p, _, crit_node = self.ctx.get_cached_stats(kicked)
            if min_p >= self.ctx.h_min:
                return kicked, True, boosts

            if not crit_node or crit_node == "ERR":
                return kicked, False, boosts

            path_pipes, _ = self.ctx.get_dominant_path(kicked, crit_node)
            if not path_pipes:
                return kicked, False, boosts

            unit_losses = self.ctx.get_cached_heuristics(kicked)
            candidates = []

            for idx in path_pipes:
                if idx in locked_pipes_local: continue

                curr_d = kicked[idx]
                if curr_d < self.ctx.max_d_idx:
                    if self.ctx.profile.is_large:
                        c_curr = self.ctx.simulator.costs[curr_d]
                        c_next = self.ctx.simulator.costs[curr_d + 1]
                        delta_c_per_m = c_next - c_curr

                        abs_cost = delta_c_per_m * self.ctx.simulator.lengths[idx]

                        efficiency = unit_losses[idx] / max(1e-6, delta_c_per_m * math.sqrt(max(1.0, abs_cost)))
                        candidates.append((idx, efficiency))
                    else:
                        candidates.append((idx, unit_losses[idx]))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_pipe = candidates[0][0]

            kicked[best_pipe] += 1

            boosts += 1

        return kicked, False, boosts

    def swap_search(self, indices, dyn_bonus):
        indices_copy = list(indices)

        raw_cost, best_p, _, crit_node = self.ctx.get_cached_stats(indices_copy)

        if not crit_node or crit_node == "ERR":
            return indices_copy

        p_surplus = best_p - self.ctx.h_min

        best_cost = self.ctx.score(raw_cost, best_p, dyn_bonus)

        path_pipes, _ = self.ctx.get_dominant_path(indices_copy, crit_node)
        unit_losses = self.ctx.get_cached_heuristics(indices_copy)
        lazy_pipes = sorted(range(self.ctx.num_pipes), key=lambda i: unit_losses[i])

        if not path_pipes or not lazy_pipes:
            return indices_copy

        down_limit = max(10, self.ctx.num_pipes // 3)

        # `200` тут була дев'ятою копією межі MEDIUM/LARGE, яку етап 8
        # звів до `NetworkProfile`. Межа та сама, поведінка не змінюється.
        if self.ctx.profile.is_large:
            down_limit = min(down_limit, 40)

        is_critically_tight = p_surplus < 0.02

        if not is_critically_tight:
            for p in lazy_pipes[:down_limit]:
                if indices_copy[p] > 0 and unit_losses[p] < 0.05:
                    test_sol = list(indices_copy)
                    test_sol[p] -= 1
                    c, p_val, _, _ = self.ctx.get_cached_stats(test_sol)
                    if p_val >= self.ctx.h_min:
                        score = self.ctx.score(c, p_val, dyn_bonus)
                        if score < best_cost:
                            best_cost, indices_copy = score, test_sol

        if not self.ctx.profile.is_large:
            safe_up_limit = min(len(path_pipes), 15)
            safe_down_limit = min(len(lazy_pipes), 20)

            for up_pipe in path_pipes[-safe_up_limit:]:
                if indices_copy[up_pipe] >= self.ctx.max_d_idx: continue

                for d1, d2 in itertools.combinations(lazy_pipes[:safe_down_limit], 2):
                    if indices_copy[d1] > 0 and indices_copy[d2] > 0 and indices_copy[up_pipe] < self.ctx.max_d_idx:
                        test_sol = list(indices_copy)
                        test_sol[up_pipe] += 1
                        test_sol[d1] -= 1
                        test_sol[d2] -= 1

                        c, p_val, _, _ = self.ctx.get_cached_stats(test_sol)
                        if p_val >= self.ctx.h_min:
                            score = self.ctx.score(c, p_val, dyn_bonus)
                            if score < best_cost:
                                best_cost, indices_copy = score, test_sol

        return indices_copy

    def evaluate_candidate(self, base_indices, pipes_to_mod, mode, dyn_bonus):
        test_sol = list(base_indices)
        locked = set()
        valid = False
        for p_idx in pipes_to_mod:
            c_idx = test_sol[p_idx]
            if mode == "upgrade" and c_idx < self.ctx.max_d_idx:
                test_sol[p_idx] += 1
                locked.add(p_idx)
                valid = True
            elif mode == "downgrade" and c_idx > 0:
                test_sol[p_idx] -= 1
                valid = True

        if not valid: return float('inf'), -float('inf'), None

        squeezed_sol = self.gradient_squeeze(test_sol, locked_pipes=locked,
                                            max_passes=self.RANKING_POLISH_PASSES,
                                            quick_mode=True, dyn_bonus=dyn_bonus)
        cost, p_min, _, _ = self.ctx.get_cached_stats(squeezed_sol)

        p_surplus = p_min - self.ctx.h_min
        self.tightness.note(p_surplus)
        if p_surplus < 0: return float('inf'), -float('inf'), None

        score = self.ctx.score(cost, p_min, dyn_bonus)
        return score, cost, squeezed_sol
