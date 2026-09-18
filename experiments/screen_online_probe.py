"""Самокалібрований відсів кандидатів — вимірювальний стенд.

Продукційного коду НЕ чіпає: усе через підміну `evaluate_candidate`.

SCREEN=off   — поведінка як у коді (контроль);
SCREEN=on    — відсів із онлайн-калібруванням.

Ідея калібрування, а не таблиці: алгоритм не може знати наперед, чи
працює дешевий прогноз на **цій** мережі. Тому він спершу дивиться, не
відсіваючи: робить прогноз, дає кандидату пройти звичайний шлях і
порівнює прогноз із правдою. Відсів вмикається лише якщо на вибірці не
було жодної хибної відмови, і лише з запасом, який виведено з
**спостереженої** похибки прогнозу. Далі частина кандидатів навмисно йде
повз відсів — щоб помітити, якщо мережа «попливла».
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import networkx as nx  # noqa: E402

from pipeopt.solver.local_search import LocalSearch  # noqa: E402

OUT = os.environ["OUT"]
SCREEN_ON = os.environ.get("SCREEN", "off") == "on"

#: Скільки кандидатів спостерігати, не відсіваючи, перед рішенням.
CALIBRATION = int(os.environ.get("CALIBRATION", "150"))
#: Частка кандидатів, які й далі йдуть повз відсів — контроль дрейфу.
AUDIT_SHARE = float(os.environ.get("AUDIT", "0.05"))
#: Показник Хазена-Вільямса для перерахунку втрат від діаметра.
DIAMETER_EXPONENT = 4.87

S = {"calls": 0, "predicted": 0, "screened_out": 0, "false_rejections": 0,
     "armed_at": None, "margin": None, "audit_checks": 0,
     "calibration_reject_share": None, "base_pressure_sims": 0}


def dump():
    with open(OUT, "w") as fh:
        json.dump(S, fh)


class Screen:
    """Стан відсіву для одного `LocalSearch`."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.paths = None
        self.base_key = None
        self.base_pressures = None
        self.samples = []          # (прогноз p_min, правда p_min)
        self.armed = False
        self.margin = 0.0

    def node_paths(self):
        """Труби на шляху від джерела до кожного вузла — лише топологія."""
        if self.paths is not None:
            return self.paths
        sources = self.ctx.simulator.sources or []
        self.paths = {}
        if not sources:
            return self.paths
        source = sources[0]
        for node in self.ctx.base_G_flow.nodes():
            try:
                route = nx.shortest_path(self.ctx.base_G_flow, source, node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            pipes = set()
            for a, b in zip(route, route[1:]):
                pipes |= (set(self.ctx.node_to_pipes.get(a, []))
                          & set(self.ctx.node_to_pipes.get(b, [])))
            self.paths[node] = pipes
        return self.paths

    def pressures_of(self, indices):
        """Тиски всіх вузлів базового рішення; одна симуляція на батька."""
        key = tuple(indices)
        if key == self.base_key:
            return self.base_pressures
        simulator = self.ctx.simulator
        try:
            simulator._apply_pattern(list(indices))
            if not simulator._run_simulation_core():
                self.base_key, self.base_pressures = key, None
                return None
            self.ctx.sim_count += 1          # чесний облік: це симуляція
            S["base_pressure_sims"] += 1
            self.base_pressures = {
                simulator.c_idx_to_node_name[c]:
                    simulator.api.ENgetnodevalue(c, 11) * simulator.press_si_mult
                for c in simulator.junction_c_indices}
        except Exception:
            self.base_pressures = None
        self.base_key = key
        return self.base_pressures

    def predict(self, base_indices, pipes_to_mod, mode):
        """Прогноз `p_min` після зміни; None — прогнозувати не вдалось."""
        if len(pipes_to_mod) != 1:
            return None                      # парні ходи не прогнозуємо
        pipe = pipes_to_mod[0]
        current = base_indices[pipe]
        new_idx = current + 1 if mode == "upgrade" else current - 1
        if not 0 <= new_idx <= self.ctx.max_d_idx:
            return None

        pressures = self.pressures_of(base_indices)
        if not pressures:
            return None
        paths = self.node_paths()
        if not paths:
            return None

        unit_losses = self.ctx.get_cached_heuristics(list(base_indices))
        headloss = unit_losses[pipe] * self.ctx.lengths[pipe]
        old_d = self.ctx.diameters[current]
        new_d = self.ctx.diameters[new_idx]
        delta_h = headloss * ((old_d / new_d) ** DIAMETER_EXPONENT - 1.0)

        worst = float("inf")
        for node, pressure in pressures.items():
            affected = pipe in paths.get(node, ())
            worst = min(worst, pressure - delta_h if affected else pressure)
        return worst

    def note(self, predicted, truth):
        """Записати пару прогноз/правда і, коли вибірки досить, вирішити."""
        self.samples.append((predicted, truth))
        if self.armed or len(self.samples) < CALIBRATION:
            return

        h_min = self.ctx.h_min
        errors = sorted(abs(p - t) for p, t in self.samples)
        # Запас — 95-й процентиль спостереженої похибки. Не константа:
        # мережа сама каже, наскільки грубий тут прогноз.
        self.margin = errors[int(0.95 * (len(errors) - 1))]

        would_reject = [(p, t) for p, t in self.samples
                        if p < h_min - self.margin]
        wrong = [1 for p, t in would_reject if t >= h_min]
        share = len(would_reject) / len(self.samples)
        S["calibration_reject_share"] = round(share, 3)
        S["margin"] = round(self.margin, 4)

        # Вмикаємо лише якщо (а) жодної хибної відмови і (б) відсів
        # узагалі щось дає. Інакше він лише додає ризик без виграшу.
        if not wrong and share >= 0.10:
            self.armed = True
            S["armed_at"] = S["calls"]


_screens = {}
_original = LocalSearch.evaluate_candidate


def wrapped(self, base_indices, pipes_to_mod, mode, dyn_bonus):
    S["calls"] += 1
    if not SCREEN_ON:
        return _original(self, base_indices, pipes_to_mod, mode, dyn_bonus)

    screen = _screens.get(id(self))
    if screen is None:
        screen = _screens[id(self)] = Screen(self.ctx)

    predicted = screen.predict(base_indices, pipes_to_mod, mode)
    if predicted is not None:
        S["predicted"] += 1

    auditing = random.random() < AUDIT_SHARE
    if (screen.armed and predicted is not None and not auditing
            and predicted < self.ctx.h_min - screen.margin):
        S["screened_out"] += 1
        return float("inf"), -float("inf"), None

    score, cost, solution = _original(self, base_indices, pipes_to_mod,
                                      mode, dyn_bonus)

    if predicted is not None:
        truth = (self.ctx.get_cached_stats(solution)[1] if solution is not None
                 else self.ctx.h_min - 1.0)
        if screen.armed:
            if auditing:
                S["audit_checks"] += 1
                if predicted < self.ctx.h_min - screen.margin and truth >= self.ctx.h_min:
                    S["false_rejections"] += 1
                    screen.armed = False      # мережа попливла — вимикаємось
        else:
            screen.note(predicted, truth)
    if S["calls"] % 200 == 0:
        dump()
    return score, cost, solution


LocalSearch.evaluate_candidate = wrapped

import main  # noqa: E402

NET = os.environ.get("NET", "Hanoi")
NETS = {"Hanoi": ("Hanoi.inp", "in", "30"),
        "Balerma": ("Balerma.inp", "mm", "20"),
        "Foss": ("Foss.inp", "mm", "40"),
        "KL": ("KL.inp", "mm", "20"),
        "UTown": ("UTown.inp", "mm", "20")}
inp, units, hmin = NETS[NET]

sys.argv = ["main.py",
            "--inp", os.path.join(ROOT, "InputData", NET, inp),
            "--costs", os.path.join(ROOT, "InputData", NET, "costs.csv"),
            "--units", units, "--hmin", os.environ.get("HMIN", hmin),
            "--cores", "1", "--sequential",
            "--seed", os.environ.get("SEED", "42"),
            "--max_sims", os.environ.get("SIMS", "150000"),
            "--time_limit", "0"]
main.main()
dump()
