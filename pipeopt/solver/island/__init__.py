"""Острів: чотири домішки й цикл, що їх з'єднує.

Острів: складання домішок і головний цикл.

`IslandWorker` зібраний із чотирьох домішок так само, як `KickStrategies`
із трьох (етап 16): диспетчер кіків викликає стратегії **за іменем**
через `getattr`, тому делегування довелося б або дублювати у фасаді, або
ускладнювати реєстр. Домішки лишають виклики тими самими, а файли —
окремими.

До розділення тут був один клас на 1210 рядків із 34 методами; шукати в
ньому потрібний метод можна було лише пошуком по тексту.
"""

from .beam import BeamSearch
from .diversification import Diversification
from .exchange import IslandExchange
from .loop import IslandWorker
from .state import AppliedKick, IslandState

__all__ = ["AppliedKick", "BeamSearch", "Diversification", "IslandExchange",
           "IslandState", "IslandWorker"]
