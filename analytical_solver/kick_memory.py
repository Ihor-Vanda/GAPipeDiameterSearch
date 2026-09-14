"""Коротка пам'ять кіків: що вже пробували і що не спрацювало.

Чотири словники, які до цього модуля жили окремими полями воркера:

* `bottleneck_failed`    — труби, на яких `BOTTLENECK` не дав результату;
* `bottleneck_boosted`   — труби, які `BOTTLENECK` уже підняв;
* `loop_balance_failed`  — те саме для `LOOP_BALANCE`;
* `zero_sum_tabu`        — труби, зайняті останнім `ZERO_SUM`.

Вони створювались разом, **очищались разом** (усі чотири в гілці
`ESCAPE`, бо після великої втечі пам'ять стосується іншого басейну) і
разом передавались у кік. Тримати їх окремими полями означало, що
додавання п'ятої пам'яті вимагає правок у чотирьох місцях, а забути
очистити одну з них у `ESCAPE` — тихий дефект.

У всіх чотирьох значення — номер раунду, коли подія сталася: кіки
використовують його як давність, щоб та сама труба не блокувалась
назавжди.
"""


class KickMemory:
    """Пам'ять кіків за стратегіями. Значення — номер раунду події."""

    #: Стратегія -> яку саме пам'ять невдач вона читає й пише.
    FAILURE_MEMORY = {
        "BOTTLENECK": "bottleneck_failed",
        "LOOP_BALANCE": "loop_balance_failed",
    }
    #: Стратегія, для якої ведеться пам'ять уже піднятих труб.
    BOOST_STRATEGY = "BOTTLENECK"
    #: Стратегія, для якої ведеться табу зайнятих труб.
    ZERO_SUM_STRATEGY = "ZERO_SUM"

    __slots__ = ("bottleneck_failed", "bottleneck_boosted",
                 "loop_balance_failed", "zero_sum_tabu")

    def __init__(self):
        self.bottleneck_failed = {}
        self.bottleneck_boosted = {}
        self.loop_balance_failed = {}
        self.zero_sum_tabu = {}

    def clear(self):
        """Забути все: після великої втечі пам'ять стосується іншого басейну."""
        for name in self.__slots__:
            getattr(self, name).clear()

    # --- читання ----------------------------------------------------------

    def failed_pipes_for(self, strategy):
        """Пам'ять невдач, яку читає ця стратегія.

        Поведінка збережена дослівно: `BOTTLENECK` читає свою, **усі
        інші** — пам'ять `LOOP_BALANCE`. Так було в тернарному виразі
        всередині збірки аргументів кіка.
        """
        name = self.FAILURE_MEMORY.get(strategy, "loop_balance_failed")
        return getattr(self, name)

    def as_kick_args(self, strategy):
        """Пам'ять у вигляді аргументів кіка."""
        return {
            "failed_pipes": self.failed_pipes_for(strategy),
            "boosted_pipes": self.bottleneck_boosted,
            "zero_sum_tabu": self.zero_sum_tabu,
        }

    # --- запис ------------------------------------------------------------

    def note_failure(self, strategy, pipe_idx, round_idx):
        """Запам'ятати трубу, на якій стратегія не дала результату.

        Пишеться лише для стратегій, які цю пам'ять читають; для решти
        запис був би марним — саме так і було в оригінальному `if/elif`.
        """
        name = self.FAILURE_MEMORY.get(strategy)
        if name is not None:
            getattr(self, name)[pipe_idx] = round_idx

    def note_boost(self, strategy, pipe_idx, round_idx):
        if strategy == self.BOOST_STRATEGY:
            self.bottleneck_boosted[pipe_idx] = round_idx

    def note_zero_sum(self, strategy, locked, round_idx):
        """Заблокувати трубу, зайняту `ZERO_SUM`, на кілька раундів."""
        if strategy == self.ZERO_SUM_STRATEGY and locked:
            self.zero_sum_tabu[next(iter(locked))] = round_idx
