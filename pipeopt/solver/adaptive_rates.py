"""Самоадаптація розміру кроку кіків (JADE-подібна).

Три кіки мають параметр «яку частку мережі перебудувати»:
`RUIN_RECREATE`, `SMART_PERTURB`, `SPATIAL_PERTURB`. Розмір кроку не
задається константою, а вчиться за результатом: кік повідомляє, яку
частку він фактично використав (`used_pct`), а середнє значення `mu`
зсувається до неї експоненційним згладжуванням — як у JADE, де параметри
мутації адаптуються за успішними випробуваннями. `BASIN_ESCAPE`
адаптується окремо й простіше.

До цього модуля ці чотири значення жили полями `IslandWorker`
(`mu_ruin_pct`, `mu_perturb_pct`, `mu_spatial_pct`, `mu_escape_pct`), а
оновлювались у **п'яти** місцях `_apply_kick`, і кожне з них повторювало
той самий ланцюг:

    if strategy == "RUIN_RECREATE":     self.mu_ruin_pct = ...
    elif strategy == "SMART_PERTURB":   self.mu_perturb_pct = ...
    elif strategy == "SPATIAL_PERTURB": self.mu_spatial_pct = ...

П'ять копій з чотирма різними альфами й двома різними політиками підлоги.
Тут ланцюг один — таблиця `STEP_SIZES`, — а п'ять місць стають п'ятьма
викликами.

Важливо: дві політики звуження **не однакові**, і це збережено дослівно.
Вибух вартості звужує крок без підлоги, звичайний відкид — з підлогою.
"""

import math


class AdaptiveRates:
    """Поточні розміри кроку і правила їх навчання."""

    #: стратегія -> (ключ значення, початкове значення, підлога при звуженні)
    STEP_SIZES = {
        "RUIN_RECREATE": ("ruin", 0.05, 0.02),
        "SMART_PERTURB": ("perturb", 0.10, 0.03),
        "SPATIAL_PERTURB": ("spatial", 0.20, 0.05),
    }

    ESCAPE_START = 0.20
    ESCAPE_ALPHA = 0.10

    #: Звуження при вибуху вартості — без підлоги (крок явно завеликий).
    EXPLOSION_SHRINK = 0.90
    #: Звуження при звичайному відкиді — з підлогою з таблиці.
    REJECT_SHRINK = 0.95

    #: Базова швидкість навчання на рекорді й межа її послаблення для
    #: великих кроків: великий `used_pct` довіряється менше, бо великий
    #: крок міг дати виграш випадково.
    RECORD_ALPHA = 0.15
    RECORD_ALPHA_DAMPING = 0.5
    #: Навчання при слабкому прийнятті (в пул) і ширина вікна близькості.
    POOL_ALPHA = 0.03
    POOL_GAP_WINDOW = 0.10

    def __init__(self):
        self.values = {key: start for key, start, _ in self.STEP_SIZES.values()}
        self.escape = self.ESCAPE_START

    # --- читання ----------------------------------------------------------

    def as_kick_args(self):
        """Розміри кроку у вигляді kwargs для кіка."""
        return {
            "mu_ruin_pct": self.values["ruin"],
            "mu_perturb_pct": self.values["perturb"],
            "mu_spatial_pct": self.values["spatial"],
            "mu_escape_pct": self.escape,
        }

    def snapshot(self):
        return dict(self.values, escape=self.escape)

    def tracks(self, strategy):
        """Чи має ця стратегія адаптивний розмір кроку."""
        return strategy in self.STEP_SIZES

    # --- альфи ------------------------------------------------------------

    @classmethod
    def record_alpha(cls, used_pct):
        """Швидкість навчання після рекорду — тим менша, чим більший крок."""
        damping = math.log1p(used_pct * 10) / math.log1p(10)
        return cls.RECORD_ALPHA * (1.0 - cls.RECORD_ALPHA_DAMPING * damping)

    @classmethod
    def pool_alpha(cls, pool_gap):
        """Швидкість навчання після прийняття в пул.

        Рішення, прийняте майже на рівні рекорду, вчить сильніше за те, що
        ледве пролізло у вікно; за межами вікна не вчить узагалі (0.0).
        """
        proximity = 1.0 - min(1.0, pool_gap / cls.POOL_GAP_WINDOW)
        return cls.POOL_ALPHA * proximity if proximity > 0 else 0.0

    # --- навчання ---------------------------------------------------------

    def learn(self, strategy, used_pct, alpha):
        """Зсунути розмір кроку стратегії до фактично використаного."""
        if strategy not in self.STEP_SIZES or not alpha:
            return
        key = self.STEP_SIZES[strategy][0]
        self.values[key] = (1 - alpha) * self.values[key] + alpha * used_pct

    def learn_escape(self, used_pct):
        self.escape = (1 - self.ESCAPE_ALPHA) * self.escape + self.ESCAPE_ALPHA * used_pct

    def shrink(self, strategy, factor, floor=False):
        """Звузити крок. `floor=True` — не нижче межі з таблиці."""
        if strategy not in self.STEP_SIZES:
            return
        key, _, minimum = self.STEP_SIZES[strategy]
        shrunk = self.values[key] * factor
        self.values[key] = max(minimum, shrunk) if floor else shrunk

    def shrink_after_explosion(self, strategy):
        self.shrink(strategy, self.EXPLOSION_SHRINK)

    def shrink_after_reject(self, strategy):
        self.shrink(strategy, self.REJECT_SHRINK, floor=True)
