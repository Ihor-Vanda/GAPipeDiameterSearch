"""Ширина раунду як **заміряна** величина, а не таблиця за класом мережі.

ЗАСТАРІЛО з етапу 27: правило переїхало в продукційний код
(`IslandState.narrow_round`), і цей стенд тепер додавав би власне
звуження **поверх** нього — «контроль» у ньому вже не контроль.
Для A/B користуйтесь `round_rule_ab.py`. Файл лишається як запис
того, чим саме правило міряли до перенесення.

Що заміряно раніше (журнал, 18.09): якщо поділити кількість кандидатів
на батька навпіл, на Балермі й на сітці 16×16 результат кращий на 6
прогонах із 6, за двох різних бюджетів. Виграш тримається на тому, що
шар диверсифікації нарешті вмикається: на сітці кіків було **нуль**, а
стало 18. На Ханої та Фоссі важіль інертний — там кандидатів і так
стільки, скільки дозволяє підлога.

Правило, яке з цього випливає, не має в собі розміру мережі:

    раунд не повинен коштувати більше за ROUND_SHARE бюджету;
    якщо коштує — звузити, поки не вкладеться.

Ціна раунду **міряється на місці**, тому правило працює на мережі
будь-якого розміру й структури, зокрема на тій, якої не було в замірі.

Правило **одностороннє**: воно лише звужує раунд і ніколи не розширює
понад те, що дає теперішня таблиця. Розширення на малих мережах ніхто не
міряв, і закладати неміряний напрямок у правило не можна — на Ханої та
Фоссі прогін мусить лишитись побітово тим самим.

    ROUND_SHARE=0.04 NET=Balerma SEED=1 SIMS=150000 \\
        python experiments/adaptive_round_probe.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver.island import beam, diversification  # noqa: E402
from pipeopt.solver.island.loop import IslandWorker  # noqa: E402

#: Частка бюджету, дорожче за яку раунд не має права коштувати.
#: 0 вимикає правило (контроль).
ROUND_SHARE = float(os.environ.get("ROUND_SHARE", "0.04"))
#: Наскільки звужувати за один крок. Не параметр задачі, а крок
#: регулятора: менший крок — повільніша збіжність, більший — перестрибує.
SHRINK_STEP = 0.7

STATE = {"single": None, "cap": None, "rounds": 0, "kicks": 0,
         "last_sims": None, "narrowed": 0}

_init = IslandWorker.__init__
_beam = beam.BeamSearch._beam_search_and_update
_kick = diversification.Diversification._apply_kick


def init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    STATE["cap"] = self.SINGLE_CANDIDATES
    STATE["single"] = self.SINGLE_CANDIDATES
    STATE["target"] = ROUND_SHARE * float(self.max_sims)


def beam_counted(self, next_gen, *args, **kwargs):
    STATE["rounds"] += 1
    before = self.ctx.sim_count
    result = _beam(self, next_gen, *args, **kwargs)
    if ROUND_SHARE > 0:
        # Ціна раунду — від початку його покоління мутацій. `next_gen`
        # уже пораховано, тому лічильник беремо з попередньої позначки.
        spent = self.ctx.sim_count - STATE.get("round_start", before)
        if spent > STATE["target"] and self.SINGLE_CANDIDATES > 6:
            self.SINGLE_CANDIDATES = max(6, int(self.SINGLE_CANDIDATES * SHRINK_STEP))
            STATE["single"] = self.SINGLE_CANDIDATES
            STATE["narrowed"] += 1
    STATE["round_start"] = self.ctx.sim_count
    return result


def mut_marked(self, *args, **kwargs):
    if "round_start" not in STATE:
        STATE["round_start"] = self.ctx.sim_count
    return _mut(self, *args, **kwargs)


_mut = beam.BeamSearch._generate_mutations


def kick_counted(self, *args, **kwargs):
    STATE["kicks"] += 1
    return _kick(self, *args, **kwargs)


IslandWorker.__init__ = init
beam.BeamSearch._beam_search_and_update = beam_counted
beam.BeamSearch._generate_mutations = mut_marked
diversification.Diversification._apply_kick = kick_counted

import main  # noqa: E402

NET = os.environ.get("NET", "Balerma")
NETS = {"Hanoi": ("Hanoi.inp", "in", "30"),
        "Foss": ("Foss.inp", "mm", "40"),
        "Balerma": ("Balerma.inp", "mm", "20"),
        "Grid16x16": ("Grid16x16.inp", "mm", "20"),
        "Grid24x24": ("Grid24x24.inp", "mm", "20")}
inp, units, hmin = NETS[NET]

sys.argv = ["main.py",
            "--inp", os.path.join(ROOT, "InputData", NET, inp),
            "--costs", os.path.join(ROOT, "InputData", NET, "costs.csv"),
            "--units", units, "--hmin", hmin, "--cores", "1", "--sequential",
            "--seed", os.environ.get("SEED", "42"),
            "--max_sims", os.environ.get("SIMS", "150000"),
            "--time_limit", "0"]
main.main()
print(f">>> ROUND_SHARE={ROUND_SHARE} стеля={STATE['cap']} "
      f"кінцева ширина={STATE['single']} звужень={STATE['narrowed']} "
      f"раундів={STATE['rounds']} кіків={STATE['kicks']}")
