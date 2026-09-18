"""Вибір стратегії кіка і статистика її успішності (UCB1 / VND).

Вибір працює у двох режимах, і перемикається між ними за температурою та
фазою прогону:

* **UCB1** (багаторукий бандит) — поки прогін «холодний» (перша половина
  бюджету) і застій помірний. Стратегія обирається за оцінкою
  `середній виграш + бонус за недовивченість`, зі штрафом за повтор.
* **VND-random** — у другій половині прогону або при сильному застої.
  Стратегія обирається випадково, але не та сама, що щойно.

Статистика ведеться **двома** незалежними таблицями — «холодною» і
«гарячою». Те, що працювало на початку прогону, не обов'язково працює в
кінці, тому в другій половині бандит вчиться з чистого аркуша.

До цього модуля все це жило в перших 55 рядках `_apply_kick`, а п'ять
словників стану читались як два через псевдоніми `active_wins` /
`active_tries`, які на початку методу вказували то на гарячу таблицю, то
на холодну. Зміни ваг були розсипані по **семи** місцях методу, і кожна
починалась із `if use_bandit`.
"""

import math
import random


class Choice:
    """Обрана стратегія і те, що про неї потрібно знати далі по методу."""

    __slots__ = ("strategy", "mode_label", "bandit_active")

    def __init__(self, strategy, mode_label, bandit_active):
        self.strategy = strategy
        self.mode_label = mode_label
        self.bandit_active = bandit_active


class StrategySelector:
    """Статистика стратегій і правило вибору наступної."""

    #: Температура, вище якої бандит вимикається на користь VND.
    BANDIT_TEMPERATURE_LIMIT = 0.60
    #: Скільки невдач поспіль виводять стратегію з обігу. При високій
    #: температурі поріг вищий: там невдачі очікувані.
    MAX_FAILS_COLD = 3
    MAX_FAILS_HOT = 5

    #: Коефіцієнт дослідження UCB1 — зростає з температурою.
    EXPLORE_BASE = 0.05
    EXPLORE_TEMPERATURE_GAIN = 0.15
    #: Штраф за повтор тієї самої стратегії: 0.5 за кожне поспіль.
    BURST_PENALTY_BASE = 0.5

    #: Межі ваги стратегії.
    WIN_CEILING = 4.0
    WIN_FLOOR = 1.0
    #: Підлога при штрафуванні — нижча за звичайну: покарана стратегія
    #: мусить мати шанс повернутись, але не швидко.
    PENALTY_FLOOR = 0.1

    FAILED_KICK_PENALTY = 0.25
    EXPLOSION_PENALTY = 0.50
    #: На рекорді решта стратегій втрачає 20 % ваги.
    OTHERS_DECAY_ON_RECORD = 0.80
    #: Після великої втечі обидві таблиці гасяться на 70 %.
    ESCAPE_DECAY = 0.3

    def __init__(self, strategies, ablation=False):
        self.ablation = ablation
        self.wins = {"cold": {s: 1.0 for s in strategies},
                     "hot": {s: 1.0 for s in strategies}}
        self.tries = {"cold": {s: 1.0 for s in strategies},
                      "hot": {s: 1.0 for s in strategies}}
        self.consecutive_fails = {s: 0.0 for s in strategies}

        self.last_strategy = None
        self.burst_count = 0

        #: Таблиця, з якою працюють `reward` / `penalise` після `pick`.
        self._phase = "cold"

    # --- доступ до таблиць ------------------------------------------------

    def phase_for(self, is_late_game):
        return "hot" if is_late_game else "cold"

    @property
    def active_wins(self):
        return self.wins[self._phase]

    @property
    def active_tries(self):
        return self.tries[self._phase]

    # --- вибір ------------------------------------------------------------

    def pick(self, pool, temperature, effective_temperature, is_late_game):
        """Обрати стратегію з `pool`.

        `temperature` — справжня T (від застою), `effective_temperature` —
        та, що бачить решта методу (в ablation-режимі вона зафіксована).
        """
        self._phase = self.phase_for(is_late_game)
        wins, tries = self.active_wins, self.active_tries

        max_fails = (self.MAX_FAILS_COLD if effective_temperature < 0.5
                     else self.MAX_FAILS_HOT)
        valid = [s for s in pool
                 if s in wins and self.consecutive_fails.get(s, 0) < max_fails]
        if not valid:
            valid = [s for s in pool if s in wins]

        untried = [s for s in pool if s not in wins]
        bandit_active = (not self.ablation and not is_late_game
                         and effective_temperature < self.BANDIT_TEMPERATURE_LIMIT)

        if untried:
            strategy = random.choice(untried)
        elif not bandit_active:
            strategy = self._pick_vnd(pool, valid)
        else:
            strategy = self._pick_ucb1(valid, wins, tries, temperature)

        self._note_pick(strategy)
        tries[strategy] = tries.get(strategy, 0) + 1

        label = (f"[UCB1 Temp: {effective_temperature:.2f}]" if bandit_active
                 else "[VND RAND]")
        return Choice(strategy, label, bandit_active)

    def _pick_vnd(self, pool, valid):
        """Випадкова стратегія, але не та сама, що щойно."""
        source = valid if valid else pool
        without_last = [s for s in source if s != self.last_strategy]
        return random.choice(without_last if without_last else source)

    def _pick_ucb1(self, valid, wins, tries, temperature):
        total = sum(tries.values())
        exploration = self.EXPLORE_BASE + self.EXPLORE_TEMPERATURE_GAIN * temperature

        def score(strategy):
            attempts = max(1, tries.get(strategy, 1))
            average_win = wins[strategy] / attempts
            explore_bonus = exploration * math.sqrt(math.log(max(1, total)) / attempts)

            burst_penalty = 1.0
            if strategy == self.last_strategy:
                burst_penalty = self.BURST_PENALTY_BASE ** self.burst_count

            return (average_win + explore_bonus) * burst_penalty

        return max(valid, key=score)

    def _note_pick(self, strategy):
        if strategy == self.last_strategy:
            self.burst_count += 1
        else:
            self.last_strategy = strategy
            self.burst_count = 1

    # --- навчання ---------------------------------------------------------

    def note_failure(self, strategy):
        self.consecutive_fails[strategy] = self.consecutive_fails.get(strategy, 0) + 1

    def note_success(self, strategy):
        self.consecutive_fails[strategy] = 0

    def reward(self, strategy, amount):
        """Підняти вагу стратегії, не вище стелі."""
        wins = self.active_wins
        wins[strategy] = min(self.WIN_CEILING, wins.get(strategy, 1.0) + amount)

    def penalise(self, strategy, factor):
        """Помножити вагу стратегії на `factor`, не нижче підлоги покарання."""
        wins = self.active_wins
        wins[strategy] = max(self.PENALTY_FLOOR, wins.get(strategy, 1.0) * factor)

    def decay_others(self, strategy, factor):
        """Приглушити решту стратегій — рекорд однієї здешевлює інші."""
        wins = self.active_wins
        for other in list(wins):
            if other != strategy:
                wins[other] = max(self.WIN_FLOOR, wins[other] * factor)

    def reset_after_escape(self):
        """Погасити обидві таблиці: попередні ваги стосувались іншого басейну."""
        for phase in ("hot", "cold"):
            for table in (self.wins[phase], self.tries[phase]):
                for strategy in table:
                    table[strategy] = max(self.WIN_FLOOR, table[strategy] * self.ESCAPE_DECAY)

    # --- нагороди, залежні від результату ---------------------------------

    @staticmethod
    def record_reward(improvement_fraction):
        """Нагорода за рекорд: 1.0 плюс 100× відносного виграшу."""
        return 1.0 + improvement_fraction * 100.0

    @staticmethod
    def diversity_reward(diversity_ratio, temperature):
        """Нагорода за різноманітність при прийнятті в пул."""
        return 0.5 * diversity_ratio * temperature
