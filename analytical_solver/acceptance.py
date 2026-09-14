"""Правило прийняття результату кіка.

Після кіка й локального пошуку рішення потрапляє в одну з п'яти
розв'язок. Сам вибір — це набір порогів, які всі залежать від
температури `T` (від застою) і фази прогону:

    RECORD      рішення дешевше за рекорд і допустиме -> новий рекорд
    ESCAPE      велика втеча з басейну -> пул скидається, рекорд адоптується
    POOL        не рекорд, але в межах вікна -> у пул («прийняття за рівнем води»)
    HAIL_MARY   при T ≈ 1: остання спроба витягнути завідомо гірше рішення
    REJECT      усе інше

Плюс три розв'язки-відмови, які обриваються раніше: `HOPELESS_PRESSURE`
(недобір тиску настільки великий, що ремонтувати нема сенсу),
`EXPLOSION` (вартість вибухнула) і `GHOST_VETO` (підозріло великий виграш
виявився артефактом симуляції).

Цей модуль **вирішує, а не діє**. Усі мутації стану воркера лишились у
`_apply_kick` дослівно в тому самому порядку: він значущий (наприклад,
порог скидання застою рахується від уже оновленого рекорду), і виносити
його означало б ризикувати непомітною зміною поведінки. Тут — тільки
пороги, у яких з'явились імена, і `classify`, у якої з'явилось одне
місце.

`is_basin_tabu` і `is_ghost` передаються **функціями**, а не значеннями:
в оригіналі вони обчислювались лише в тих гілках, куди доходив `elif`, а
`is_ghost_solution` ще й виконує симуляції. Викликати їх завчасно
означало б змінити кількість симуляцій у прогоні.
"""


class Verdict:
    """Що робити з результатом кіка."""

    __slots__ = ("kind", "gap")

    #: Розв'язки, які змінюють стан.
    RECORD = "RECORD"
    ESCAPE = "ESCAPE"
    POOL = "POOL"
    HAIL_MARY = "HAIL_MARY"
    REJECT = "REJECT"

    #: Розв'язки-відмови: далі нічого не відбувається.
    HOPELESS_PRESSURE = "HOPELESS_PRESSURE"
    EXPLOSION = "EXPLOSION"
    GHOST_VETO = "GHOST_VETO"
    UNUSABLE = "UNUSABLE"

    def __init__(self, kind, gap=0.0):
        self.kind = kind
        self.gap = gap

    def __repr__(self):
        return f"Verdict({self.kind})"

    def __eq__(self, other):
        return self.kind == (other.kind if isinstance(other, Verdict) else other)

    def __hash__(self):
        return hash(self.kind)


class AcceptancePolicy:
    """Пороги прийняття і класифікація результату."""

    # --- вибух вартості: best × (1.5 + 0.5·T) -----------------------------
    EXPLOSION_BASE = 1.5
    EXPLOSION_TEMPERATURE_GAIN = 0.5

    # --- вікно пулу: best × (1.03 + 0.07·T) -------------------------------
    POOL_WINDOW_BASE = 1.03
    POOL_WINDOW_TEMPERATURE_GAIN = 0.07

    #: Межі розриву для великих мереж не діяли й **видалені** за заміром:
    #: на Балермі розрив прийнятих у пул рішень не перевищив 5.4 %, а понад
    #: 8 % не було жодного (§21.4). Причина структурна — вікно прийняття
    #: саме собою тугіше за найменшу з тих меж, і від розміру мережі не
    #: залежить, тому й на більших мережах картина та сама.

    #: Температура, з якої дозволена остання спроба.
    HAIL_MARY_TEMPERATURE = 0.95

    #: Скільки метрів недобору тиску ще терпиться — пропорційно температурі.
    DEFICIT_TOLERANCE_PER_T = 0.5
    #: Недобір, після якого ремонт безнадійний: max(20 м, h_min × (1 + T)).
    HOPELESS_PRESSURE_FLOOR = 20.0
    #: Недобір конвертується у вартість, щоб порівнюватись із рекордом.
    DEFICIT_COST_PENALTY = 0.10

    #: Виграш понад цю частку рекорду перевіряється на артефакт симуляції.
    GHOST_VETO_FRACTION = 0.02

    #: Вікно «серцебиття» для глибокого поліру: 0.03 на початку, 0.01 у кінці.
    HEARTBEAT_MARGIN_BASE = 0.03
    HEARTBEAT_MARGIN_PROGRESS_GAIN = 0.02

    def __init__(self, ablation=False):
        self.ablation = ablation

    # --- температура ------------------------------------------------------

    def effective_temperature(self, real_temperature):
        """T, яку бачить решта методу. В ablation-режимі зафіксована."""
        return 0.5 if self.ablation else real_temperature

    # --- пороги -----------------------------------------------------------

    def explosion_threshold(self, best_cost, temperature):
        return best_cost * (self.EXPLOSION_BASE
                            + self.EXPLOSION_TEMPERATURE_GAIN * temperature)

    def pool_window(self, best_cost, temperature):
        return best_cost * (self.POOL_WINDOW_BASE
                            + self.POOL_WINDOW_TEMPERATURE_GAIN * temperature)

    def deficit_tolerance(self, temperature):
        """Скільки недобору тиску ще приймається."""
        if self.ablation:
            return 0.0
        return self.DEFICIT_TOLERANCE_PER_T * temperature

    def relaxed_h_min(self, h_min, temperature):
        """Планка тиску, нижче якої запускається ремонт мережі."""
        return h_min - self.deficit_tolerance(temperature)

    def is_hopeless_pressure(self, h_min, p_min, temperature):
        """Недобір настільки великий, що ремонтувати нема сенсу."""
        deficit = max(0.0, h_min - p_min)
        return deficit > max(self.HOPELESS_PRESSURE_FLOOR,
                             h_min * (1.0 + temperature))

    def effective_cost(self, cost, deficit, best_cost):
        """Вартість із недобором тиску, переведеним у гроші."""
        return cost + deficit * (best_cost * self.DEFICIT_COST_PENALTY)

    def heartbeat_margin(self, progress_ratio):
        return (self.HEARTBEAT_MARGIN_BASE
                - self.HEARTBEAT_MARGIN_PROGRESS_GAIN * progress_ratio)

    # --- глибина поліру ---------------------------------------------------

    def deep_passes(self, gap, temperature, profile):
        """Скільки проходів глибокого спуску варті цього рішення.

        Гаряче — мало (рішення все одно зміниться); холодне і краще за
        рекорд — багато, бо це кандидат на запис. Глибину для великих
        мереж дає `NetworkProfile`, а не власний поріг 200 труб.
        """
        if temperature >= 0.6:
            return 1
        if temperature >= 0.4:
            return 2
        if gap < -0.0001:
            return profile.deep_polish_passes
        if gap <= 0.02:
            return 3 if gap <= 0.002 else 2
        return 2 if gap <= 0.05 else 1

    # --- класифікація -----------------------------------------------------

    def classify(self, *, is_escape, effective_cost, deficit, best_cost,
                 temperature, is_basin_tabu, is_ghost):
        """Обрати розв'язку. `is_basin_tabu` і `is_ghost` — функції (див. модуль)."""
        if deficit == 0 and effective_cost < best_cost:
            gain = best_cost - effective_cost
            if gain > best_cost * self.GHOST_VETO_FRACTION and is_ghost():
                return Verdict(Verdict.GHOST_VETO)
            return Verdict(Verdict.RECORD)

        if is_escape and deficit == 0:
            return Verdict(Verdict.ESCAPE)

        if effective_cost < self.pool_window(best_cost, temperature) and not is_basin_tabu():
            gap = (effective_cost - best_cost) / max(best_cost, 1)
            return Verdict(Verdict.POOL, gap)

        if temperature >= self.HAIL_MARY_TEMPERATURE and not is_basin_tabu():
            return Verdict(Verdict.HAIL_MARY)

        return Verdict(Verdict.REJECT)
