"""Контракт і реєстр стратегій диверсифікації (кіків).

До цього модуля кожен із десяти методів `KickStrategies` повертав кортеж
різної довжини, а четвертий елемент означав різне залежно від стратегії —
і `IslandWorker._apply_kick` розбирав це двадцятьма рядками if/elif. Тепер
контракт один (`KickResult`), а перелік стратегій, їхні методи й
температурні пули описані даними, а не ланцюгом з одинадцяти `elif`.
"""

from dataclasses import dataclass, field


@dataclass
class KickResult:
    """Результат застосування однієї стратегії диверсифікації.

    `solution` і `locked` разом означають успіх: якщо будь-яке з них None,
    кік не дав нічого (`failed`). Решта полів — необов'язкові дані, які
    потрібні лише окремим стратегіям:

    * `failed_pipe` — BOTTLENECK і LOOP_BALANCE повідомляють трубу, на якій
      кік не вдався, щоб воркер міг занести її в tabu;
    * `path_signature` — TOPO_INV повідомляє підпис альтернативного шляху;
    * `used_pct` — стратегії з самоадаптацією розміру (JADE) повідомляють,
      яку частку мережі фактично збурили.
    """

    solution: list | None = None
    locked: set | None = None
    message: str = ""
    failed_pipe: int | None = None
    path_signature: tuple | None = None
    used_pct: float | None = None

    @property
    def failed(self) -> bool:
        return self.solution is None or self.locked is None

    @classmethod
    def failure(cls, message="", failed_pipe=None):
        return cls(solution=None, locked=None, message=message, failed_pipe=failed_pipe)


@dataclass
class KickArgs:
    """Усе, що диспетчер передає стратегії диверсифікації.

    До цього типу `_apply_kick` збирав словник на дванадцять ключів і
    віддавав його всім кікам однаково, а кожна стратегія розбирала його
    вручну: `kwargs.get('failed_pipes', {})`. Ціна такого контракту —
    опечатка в ключі мовчки давала дефолт, а щоб побачити, які аргументи
    стратегія справді читає, доводилось читати її тіло.

    Тепер аргументи названі один раз, і `args.failed_pipes` перевіряється
    редактором ще до запуску. Це закриває борг, чесно записаний у
    коментарі `forcing_hand_kick` після етапу 3.

    Дефолти існують для тестів: стратегію можна викликати з `KickArgs()`
    і не перелічувати десять полів, які їй не потрібні.
    """

    #: «Температура відчаю»: 0 — щойно був прогрес, 1 — глибокий застій.
    T: float = 0.5

    # --- коротка пам'ять кіків (`KickMemory`) ---------------------------
    #: Труби, на яких стратегія вже не спрацювала: {труба: раунд}.
    failed_pipes: dict = field(default_factory=dict)
    #: Труби, які BOTTLENECK уже піднімав.
    boosted_pipes: dict = field(default_factory=dict)
    #: Труби, зайняті останнім ZERO_SUM.
    zero_sum_tabu: dict = field(default_factory=dict)
    #: Номер поточного раунду — пам'ять зберігає давність, а не факт.
    current_round: int = 0

    # --- стан пошуку ----------------------------------------------------
    #: Підписи шляхів, які TOPO_INV уже пробував.
    tabu_set: set = field(default_factory=set)
    #: Премія за запас тиску — потрібна кікам, що самі рахують score.
    dyn_bonus: float = 0.0
    #: Найкращі різні рішення островів: донори для BASIN_ESCAPE.
    global_archive: list = field(default_factory=list)
    #: Стеля збурення в трубах.
    max_perturb: int = 0

    # --- розміри кроку, які стратегії навчають за JADE (`AdaptiveRates`) --
    mu_ruin_pct: float = 0.05
    mu_perturb_pct: float = 0.10
    mu_spatial_pct: float = 0.20
    mu_escape_pct: float = 0.20


@dataclass(frozen=True)
class KickSpec:
    """Опис однієї стратегії.

    `adaptive_rate` — імʼя коефіцієнта μ, який ця стратегія навчає за JADE
    (`ruin`, `perturb`, `spatial`, `escape`); None — стратегія розміру не
    адаптує. `needs_cycles` — стратегія безсенсова на деревоподібній мережі.
    `extra` — що саме стратегія кладе в необовʼязкові поля KickResult.
    """

    name: str
    method: str
    needs_cycles: bool = False
    adaptive_rate: str | None = None
    extra: tuple = field(default_factory=tuple)


KICK_REGISTRY = {
    s.name: s
    for s in (
        KickSpec("SHOCK", "forcing_hand_kick"),
        KickSpec("BOTTLENECK", "upstream_bottleneck_kick", extra=("failed_pipe",)),
        KickSpec("TOPO_INV", "topological_inversion_kick", extra=("path_signature",)),
        KickSpec("LOOP_BALANCE", "loop_balancing_kick", needs_cycles=True, extra=("failed_pipe",)),
        KickSpec("ZERO_SUM", "zero_sum_shift_kick"),
        KickSpec("TRIM", "peripheral_trim_kick"),
        KickSpec("SMART_PERTURB", "smart_perturbation_kick", adaptive_rate="perturb", extra=("used_pct",)),
        KickSpec("SPATIAL_PERTURB", "spatial_perturb_kick", adaptive_rate="spatial", extra=("used_pct",)),
        KickSpec("RUIN_RECREATE", "ruin_and_recreate_kick", adaptive_rate="ruin", extra=("used_pct",)),
        KickSpec("BASIN_ESCAPE", "basin_escape", adaptive_rate="escape", extra=("used_pct",)),
    )
}

#: Стратегія за замовчуванням, якщо в реєстрі раптом немає потрібної.
FALLBACK_KICK = "SHOCK"

# Температурні пули. Це три якісно різні набори, а не діапазони [min, max]
# на кожну стратегію: у гарячому режимі набір змінюється цілком, а не
# доповнюється. Саме тому пули описані переліком, а не межами.
COLD_POOL = ("SHOCK", "BOTTLENECK", "LOOP_BALANCE", "ZERO_SUM", "TRIM")
WARM_POOL = COLD_POOL + ("SPATIAL_PERTURB", "SMART_PERTURB", "RUIN_RECREATE", "TOPO_INV")
HOT_POOL_WITH_ARCHIVE = ("BASIN_ESCAPE",)
HOT_POOL_NO_ARCHIVE = ("RUIN_RECREATE", "SPATIAL_PERTURB", "TOPO_INV")

WARM_THRESHOLD = 0.40
HOT_THRESHOLD = 0.90

#: Усі стратегії, для яких ведеться статистика UCB1.
TRACKED_STRATEGIES = tuple(KICK_REGISTRY)


def pool_for(temperature, archive_size, is_tree):
    """Які стратегії доступні за поточної «температури відчаю».

    Чиста функція — тестується без симулятора.
    """
    if temperature >= HOT_THRESHOLD:
        pool = HOT_POOL_WITH_ARCHIVE if archive_size >= 2 else HOT_POOL_NO_ARCHIVE
    elif temperature >= WARM_THRESHOLD:
        pool = WARM_POOL
    else:
        pool = COLD_POOL

    if is_tree:
        pool = tuple(s for s in pool if not KICK_REGISTRY[s].needs_cycles)
    return list(pool)
