"""Результат одного гідравлічного розрахунку.

Окремий модуль без залежностей: це предметний тип задачі, а не частина
адаптера до EPANET. Завдяки цьому солвер і тести можуть працювати з ним
без імпорту wntr.
"""

from typing import NamedTuple


class SimStats(NamedTuple):
    """Вартість рішення та тиски, отримані з гідравлічного розрахунку.

    Це кортеж, тому старе розпакування `c, p, _, node = get_stats(x)`
    працює без змін. Іменовані поля прибирають головне джерело плутанини:
    третій елемент — це МАКСИМАЛЬНИЙ тиск у мережі, а не ознака
    допустимості. Раніше його повсюдно розпаковували в змінну `feas` і
    перевіряли як `if feas`, що майже завжди істинне — у разі збою
    симуляції там -1.0, тобто теж truthy.
    """

    cost: float
    p_min: float
    p_max: float
    crit_node: str

    @property
    def simulation_ok(self) -> bool:
        """False, якщо гідравлічний розрахунок не зійшовся."""
        return self.crit_node != "ERR"

    def is_feasible(self, h_min: float) -> bool:
        return self.simulation_ok and self.p_min >= h_min

    def surplus(self, h_min: float) -> float:
        """Запас тиску над обмеженням; від'ємний — дефіцит."""
        return self.p_min - h_min
