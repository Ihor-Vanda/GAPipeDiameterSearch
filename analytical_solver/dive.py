"""SUBMARINE — стратегічна осциляція навколо межі допустимості.

Задум: басейни допустимих рішень у задачі вибору діаметрів розділені
недопустимими перешийками, тому пройти між ними, не порушивши обмеження
тиску ні на крок, часто неможливо. Механізм тимчасово опускає вимогу,
дає пошуку пройти «глибше», а потім **обов'язково** повертає вимогу, щоб
рішення виплило в новій точці простору.

У коді до цього модуля послаблення вже існувало в чотирьох місцях з
чотирма незалежними числами (0.5·T у `_apply_kick`, фіксовані −0.5 м у
beam, два різні штрафи), але без двох ключових речей: фази випливання й
точки відкату. Тут вони є.

Режими (`FeasibilityPolicy.mode`):

* `off`   — додаткового послаблення немає. Це **точне** відтворення
            поведінки до цього модуля: `extra_tolerance()` завжди 0.
* `relax` — стала додаткова поступка `depth` метрів, без фази випливання
            й без відкату. Це навмисно «неповний» варіант: він потрібен,
            щоб відокремити вплив самого послаблення від впливу
            дисциплінованого циклу занурення.
* `dive`  — повний цикл: занурення за застоєм, обмежена тривалість,
            примусове випливання, відкат у разі невдачі.

Порівняння `off` / `relax` / `dive` відповідає на питання, чи саме
відсутність випливання ламала попередню реалізацію.
"""

from dataclasses import dataclass, field

FEASIBILITY_MODES = ("off", "relax", "dive")

#: Звідки випливати. `cheapest` — найдешевше рішення пулу; воно ж і
#: найглибше занурене, тобто найближче до того самого локального
#: оптимуму, і `heal_network` піднімає його тим самим домінантним
#: шляхом. `farthest` — найвіддаленіше за Хеммінгом від точки занурення.
SURFACE_SOURCES = ("cheapest", "farthest")


@dataclass
class DiveStats:
    """Телеметрія для оцінки перспективності механізму."""

    dives: int = 0
    surfaced: int = 0            # виплили у ДОПУСТИМЕ рішення
    improved: int = 0            # випливання САМЕ побило поточний рекорд
    passive: int = 0             # виплили дешевше за точку занурення, але
                                 # рекорд встиг покращити звичайний пошук —
                                 # заслуга не занурення
    rolled_back: int = 0         # відкат: або не виплили, або не краще
    total_gain: float = 0.0      # сумарний виграш, що належить зануренням

    def as_log(self):
        return (f"dives={self.dives} surfaced={self.surfaced} "
                f"improved={self.improved} passive={self.passive} "
                f"rollback={self.rolled_back} gain={self.total_gain/1e6:.4f}M$")


@dataclass
class FeasibilityPolicy:
    """Володіє додатковою поступкою за тиском і фазами занурення.

    Уся поступка проходить через `extra_tolerance()`. Це єдине джерело —
    саме тому режим `off` гарантовано не змінює поведінку.
    """

    mode: str = "off"
    depth: float = 0.5           # метрів нижче h_min
    max_rounds: int = 8          # скільки раундів триває занурення
    cooldown_rounds: int = 12    # пауза між зануреннями
    surface_from: str = "cheapest"
    surface_attempts: int = 1    # скільки кандидатів пробувати підняти
    trigger_multiple: float = 2.0  # занурюємось при stagnation >= stag_limit * це

    diving: bool = False
    rounds_under: int = 0
    last_surfaced_round: int = -10 ** 9
    surface_cost: float = float("inf")
    surface_sol: list | None = None
    stats: DiveStats = field(default_factory=DiveStats)

    def __post_init__(self):
        if self.mode not in FEASIBILITY_MODES:
            raise ValueError(f"mode={self.mode!r}; очікується одне з {FEASIBILITY_MODES}")
        if self.surface_from not in SURFACE_SOURCES:
            raise ValueError(f"surface_from={self.surface_from!r}; "
                             f"очікується одне з {SURFACE_SOURCES}")
        if self.surface_attempts < 1:
            raise ValueError("surface_attempts мусить бути >= 1")

    # --- поступка ---------------------------------------------------------

    def extra_tolerance(self):
        """Скільки метрів дефіциту додатково дозволено просто зараз."""
        if self.mode == "relax":
            return self.depth
        if self.mode == "dive" and self.diving:
            return self.depth
        return 0.0

    # --- цикл занурення ---------------------------------------------------

    def maybe_begin(self, round_idx, stagnation, stag_limit, best_cost, best_sol):
        """Занурюємось, якщо пошук застряг і кулдаун вичерпано."""
        if self.mode != "dive" or self.diving or best_sol is None:
            return False
        if round_idx - self.last_surfaced_round < self.cooldown_rounds:
            return False
        if stagnation < max(1, stag_limit) * self.trigger_multiple:
            return False

        self.diving = True
        self.rounds_under = 0
        self.surface_cost = best_cost
        self.surface_sol = list(best_sol)
        self.stats.dives += 1
        return True

    def tick(self):
        if self.diving:
            self.rounds_under += 1

    def should_surface(self):
        return self.diving and self.rounds_under >= self.max_rounds

    def note_surfaced(self, round_idx, feasible, new_cost, current_best):
        """Закриває занурення й веде чесний облік заслуги.

        Під час занурення звичайний пошук продовжує працювати й може сам
        покращити рекорд строго допустимим шляхом. Тому порівняння з
        точкою занурення (`surface_cost`) заслугу занурення **завищує** —
        зарахувати її можна лише тоді, коли випливання побило рекорд,
        актуальний саме на момент випливання (`current_best`).
        """
        self.diving = False
        self.rounds_under = 0
        self.last_surfaced_round = round_idx

        if not feasible:
            self.stats.rolled_back += 1
            return False

        self.stats.surfaced += 1
        if new_cost < current_best:
            self.stats.improved += 1
            self.stats.total_gain += current_best - new_cost
            return True

        if new_cost < self.surface_cost:
            self.stats.passive += 1
        self.stats.rolled_back += 1
        return False
