"""Ціна внутрішньої поліровки кандидата — вимірювальний стенд.

`evaluate_candidate` полірує **кожного** кандидата
`gradient_squeeze(max_passes=3)`. На Балермі це 222 симуляції на виклик і
24 625 на раунд; саме звідси береться те, що за 200 тис. симуляцій
відбувається десять раундів.

PASSES=3  — як у коді (контроль);
PASSES=1  — один прохід;
PASSES=adaptive — проходи виводяться з розміру мережі так, щоб ціна
    кандидата лишалась співмірною з малою мережею.

Жодного продукційного коду не змінює. RNG не чіпає: підміна не тягне
випадковості, тому при PASSES=3 траєкторія має збігтися з контролем
побітово — це й перевіряється першим прогоном.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver.local_search import LocalSearch  # noqa: E402

OUT = os.environ["OUT"]
MODE = os.environ.get("PASSES", "3")

S = {"calls": 0, "passes_used": None, "pipes": 0}


def dump():
    with open(OUT, "w") as fh:
        json.dump(S, fh)


_original = LocalSearch.gradient_squeeze
_evaluate = LocalSearch.evaluate_candidate


def passes_for(ctx):
    """Скільки проходів давати внутрішній поліровці.

    `adaptive` тримає ціну кандидата приблизно сталою: один прохід
    коштує O(n) симуляцій, тому на мережі вдесятеро більшій за Ханої
    проходів має бути менше, а не стільки ж. Формула навмисно проста —
    це замір, а не готова евристика.
    """
    if MODE == "adaptive":
        if ctx.num_pipes <= 60:
            return 3
        if ctx.num_pipes <= 200:
            return 2
        return 1
    return int(MODE)


#: Позначка: виклик іде саме з `evaluate_candidate`, а не звідкись іще.
_inside = {"flag": False}


def evaluate(self, base_indices, pipes_to_mod, mode, dyn_bonus):
    S["calls"] += 1
    S["pipes"] = self.ctx.num_pipes
    if S["passes_used"] is None:
        S["passes_used"] = passes_for(self.ctx)
    _inside["flag"] = True
    try:
        return _evaluate(self, base_indices, pipes_to_mod, mode, dyn_bonus)
    finally:
        _inside["flag"] = False


def squeeze(self, *args, **kwargs):
    # `*args/**kwargs` навмисно: у справжньої `gradient_squeeze` сім
    # параметрів, і `SeedFactory` викликає її з `classical_descent=True`.
    # Перша версія підміни перелічувала параметри вручну, ковтала два з
    # них — зерна падали з TypeError, прогін тихо відкочувався на
    # запасне рішення (21.64 M$ замість 2.14). Помилка стенда, не коду.
    if _inside["flag"] and kwargs.get("max_passes") == 3:
        kwargs["max_passes"] = passes_for(self.ctx)
    if S["calls"] % 500 == 0:
        dump()
    return _original(self, *args, **kwargs)


LocalSearch.evaluate_candidate = evaluate
LocalSearch.gradient_squeeze = squeeze

import main  # noqa: E402

NET = os.environ.get("NET", "Hanoi")
NETS = {"Hanoi": ("Hanoi.inp", "in", "30"),
        "Balerma": ("Balerma.inp", "mm", "20"),
        "Foss": ("Foss.inp", "mm", "40"),
        "KL": ("KL.inp", "mm", "20"),
        "Grid24x24": ("Grid24x24.inp", "mm", "20"),
        "Grid16x16": ("Grid16x16.inp", "mm", "20")}
inp, units, hmin = NETS[NET]

sys.argv = ["main.py",
            "--inp", os.path.join(ROOT, "InputData", NET, inp),
            "--costs", os.path.join(ROOT, "InputData", NET, "costs.csv"),
            "--units", units, "--hmin", hmin, "--cores", "1", "--sequential",
            "--seed", os.environ.get("SEED", "42"),
            "--max_sims", os.environ.get("SIMS", "150000"),
            "--time_limit", "0"]
main.main()
dump()
