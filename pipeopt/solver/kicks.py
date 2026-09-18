"""Стратегії диверсифікації: один клас, зібраний із трьох тематичних груп.

`kicks.py` виріс до 543 рядків із десятьма методами й індексом
підтримуваності 0.23 — читати його цілком було ні до чого, бо групи
стратегій не мають між собою нічого спільного, крім контракту.

Розділення зроблено домішками, а не окремими об'єктами, свідомо:
диспетчер у `_apply_kick` викликає стратегію **за іменем методу** з
реєстру (`getattr(self.kicker, spec.method)`), і будь-яке делегування
довелося б або дублювати в фасаді, або ускладнювати реєстр. Домішки
лишають виклик тим самим, а файли — окремими.

Групи:

* `kicks_path` — ходи вздовж домінантного шляху до критичного вузла;
* `kicks_structural` — ходи від топології й вартості;
* `kicks_ruin` — зруйнувати частину рішення й відбудувати (самоадаптивні).
"""

from .kicks_path import PathKicks
from .kicks_ruin import RuinKicks
from .kicks_structural import StructuralKicks


class KickStrategies(PathKicks, StructuralKicks, RuinKicks):
    """Усі десять стратегій з єдиним контрактом `(indices, KickArgs) -> KickResult`."""

    def __init__(self, ctx, local_search):
        self.ctx = ctx
        self.ls = local_search
