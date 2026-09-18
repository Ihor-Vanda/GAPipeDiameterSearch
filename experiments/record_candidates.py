"""Записати кандидатів, які промінь оцінював у справжньому прогоні.

Стенд нічого не змінює: обгортка лише **дивиться** на аргументи
`evaluate_candidate` і складає їх у файл. Далі окремий стенд програє їх
наново з різною глибиною поліровки — уже не втручаючись у прогін.

Розділення на «записати» і «програти» тут не з охайності, а з потреби:
рахувати ціну проходів **усередині** живого пошуку не можна, бо кеш
симуляцій уже прогрітий попередніми кандидатами, і другий замір коштує
не стільки, скільки коштував би перший.

NET, SEED, SIMS, ROUNDS, OUT — змінні середовища.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver.island import IslandWorker  # noqa: E402
from pipeopt.solver.local_search import LocalSearch  # noqa: E402

OUT = os.environ["OUT"]
ROUNDS = int(os.environ.get("ROUNDS", "12"))

STATE = {"round": 0}
RECORDS = []

_evaluate = LocalSearch.evaluate_candidate
_beam = IslandWorker._beam_search_and_update


def evaluate(self, base_indices, pipes_to_mod, mode, dyn_bonus):
    if STATE["round"] < ROUNDS:
        RECORDS.append({"round": STATE["round"],
                        "base": list(base_indices),
                        "pipes": list(pipes_to_mod),
                        "mode": mode,
                        "dyn_bonus": dyn_bonus})
    return _evaluate(self, base_indices, pipes_to_mod, mode, dyn_bonus)


def beam(self, next_gen, just_flushed, shared_progress, round_idx):
    STATE["round"] = round_idx + 1
    STATE["beam_width"] = self.BEAM_WIDTH
    STATE["pipes"] = self.ctx.num_pipes
    if round_idx + 1 >= ROUNDS:
        dump()
    return _beam(self, next_gen, just_flushed, shared_progress, round_idx)


def dump():
    with open(OUT, "w") as fh:
        json.dump({"beam_width": STATE.get("beam_width"),
                   "pipes": STATE.get("pipes"),
                   "candidates": RECORDS}, fh)


LocalSearch.evaluate_candidate = evaluate
IslandWorker._beam_search_and_update = beam

import main  # noqa: E402

NET = os.environ.get("NET", "Hanoi")
NETS = {"Hanoi": ("Hanoi.inp", "in", "30"),
        "Balerma": ("Balerma.inp", "mm", "20"),
        "Foss": ("Foss.inp", "mm", "40")}
inp, units, hmin = NETS[NET]

sys.argv = ["main.py",
            "--inp", os.path.join(ROOT, "InputData", NET, inp),
            "--costs", os.path.join(ROOT, "InputData", NET, "costs.csv"),
            "--units", units, "--hmin", hmin, "--cores", "1", "--sequential",
            "--seed", os.environ.get("SEED", "42"),
            "--max_sims", os.environ.get("SIMS", "60000"),
            "--time_limit", "0"]
main.main()
dump()
