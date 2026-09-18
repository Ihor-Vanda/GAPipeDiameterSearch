"""Чи краще багато дешевих раундів, ніж мало дорогих.

Це **спільна передумова** обох стратегічних гілок, і доти вона не
перевірялась. Замір показав, що на Балермі за 165 тис. симуляцій
відбувається 15 раундів і 3 кіки: шар диверсифікації там майже не
працює, бо раунд коштує надто багато. Звідси дві протилежні дороги —
«зробити раунд дешевим, щоб кіки ожили» і «прийняти, що на великій
мережі це чистий спуск». Обидві спираються на одне й те саме
припущення: що від більшої кількості дешевших раундів стає краще.

Стенд перевіряє припущення **без жодного нового механізму**: він просто
ділить кількість кандидатів на раунд на `SHRINK`. Раунд стає дешевшим у
стільку ж разів, раундів за той самий бюджет стає більше. Якщо якість не
покращується — гілка «зробити раунд дешевим» відпадає, і жодної
пер-трубної пам'яті врожаю будувати не треба.

    SHRINK=4 NET=Balerma SEED=1 SIMS=150000 \\
        python experiments/round_size_probe.py

`SHRINK=1` — контроль (мусить збігтися з еталонним прогоном побітово).
Друкує додатково кількість раундів і кіків, бо саме вони й мають зрости.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver.island import beam, diversification  # noqa: E402
from pipeopt.solver.island.loop import IslandWorker  # noqa: E402

SHRINK = int(os.environ.get("SHRINK", "1"))
#: KICKS=off вимикає шар диверсифікації. Вужчий раунд дає одночасно два
#: ефекти — більше кіків і дешевшого кандидата, — і без цього вимикача
#: не сказати, який із них тримає виграш.
KICKS = os.environ.get("KICKS", "on")
COUNT = {"rounds": 0, "kicks": 0, "candidates": 0, "single": None}

_init = IslandWorker.__init__


def init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    if SHRINK > 1:
        # `MIN_CANDIDATES` профілю — 6; нижче не опускаємось, інакше
        # порівнювались би не «дешевші раунди», а «раунди без кандидатів».
        self.SINGLE_CANDIDATES = max(6, self.SINGLE_CANDIDATES // SHRINK)
    COUNT["single"] = self.SINGLE_CANDIDATES


_beam = beam.BeamSearch._beam_search_and_update
_mut = beam.BeamSearch._generate_mutations
_kick = diversification.Diversification._apply_kick


def beam_counted(self, next_gen, *args, **kwargs):
    COUNT["rounds"] += 1
    return _beam(self, next_gen, *args, **kwargs)


def mut_counted(self, *args, **kwargs):
    result = _mut(self, *args, **kwargs)
    COUNT["candidates"] += len(result)
    return result


def kick_counted(self, *args, **kwargs):
    COUNT["kicks"] += 1
    return _kick(self, *args, **kwargs)


IslandWorker.__init__ = init
if KICKS == "off":
    diversification.Diversification._apply_kick = lambda self, *a, **k: None
beam.BeamSearch._beam_search_and_update = beam_counted
beam.BeamSearch._generate_mutations = mut_counted
if KICKS != "off":
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
print(f">>> SHRINK={SHRINK} кандидатів/батька={COUNT['single']} "
      f"раундів={COUNT['rounds']} кіків={COUNT['kicks']} "
      f"кандидатів усього={COUNT['candidates']} кіки={KICKS}")
