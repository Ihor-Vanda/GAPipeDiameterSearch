"""Скільки бюджету з'їдає стадія насіння — вимірювальний стенд.

Питання виникло з етапу 25, з хибної здогадки: я вирішив був, що на
сітці 24×24 прогін не доходить до променя, бо насіння з'їдає бюджет.
Здогадка не підтвердилась (див. виправлення в журналі за 17.09), але
питання лишилось правильне — **скільки саме коштує насіння** і як ця
частка росте з мережею.

Стенд не змінює нічого: він лише запам'ятовує лічильник симуляцій на
**першому** виклику `evaluate_candidate` — усе, що витрачено до нього, і
є ціною насіння.

NET, SEED, SIMS, OUT — змінні середовища.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver.local_search import LocalSearch  # noqa: E402

OUT = os.environ["OUT"]
S = {"seed_sims": None, "total_sims": 0, "candidates": 0, "pipes": 0,
     "longest_squeeze": 0}

_evaluate = LocalSearch.evaluate_candidate
_squeeze = LocalSearch.gradient_squeeze


def dump():
    with open(OUT, "w") as fh:
        json.dump(S, fh)


def evaluate(self, *args, **kwargs):
    if S["seed_sims"] is None:
        S["seed_sims"] = self.ctx.sim_count
        S["pipes"] = self.ctx.num_pipes
        dump()
    S["candidates"] += 1
    S["total_sims"] = self.ctx.sim_count
    if S["candidates"] % 200 == 0:
        dump()
    return _evaluate(self, *args, **kwargs)


def squeeze(self, *args, **kwargs):
    # Найдорожчий одиничний виклик спуску: саме він на 1105 трубах
    # тримав процес понад 54 хвилини.
    before = self.ctx.sim_count
    try:
        return _squeeze(self, *args, **kwargs)
    finally:
        spent = self.ctx.sim_count - before
        if spent > S["longest_squeeze"]:
            S["longest_squeeze"] = spent
            S["total_sims"] = self.ctx.sim_count


LocalSearch.evaluate_candidate = evaluate
LocalSearch.gradient_squeeze = squeeze

import main  # noqa: E402

NET = os.environ.get("NET", "Hanoi")
NETS = {"Hanoi": ("Hanoi.inp", "in", "30"),
        "Balerma": ("Balerma.inp", "mm", "20"),
        "Foss": ("Foss.inp", "mm", "40"),
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
dump()
