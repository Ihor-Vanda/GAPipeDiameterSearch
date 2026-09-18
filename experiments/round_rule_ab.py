"""A/B правила ширини раунду **на продукційному коді**.

Етап 27 переніс правило з стенда в солвер, тому старий
`adaptive_round_probe.py` більше не годиться для A/B: він додавав власне
звуження **поверх** продукційного, і «контроль» у ньому вже не був
контролем. Цей стенд нічого не додає — він лише вимикає правило або
міняє його поріг.

    ROUND_SHARE=off  — правило вимкнене (контроль);
    ROUND_SHARE=0.04 — як у коді;
    ROUND_SHARE=0.02 — інший поріг.

Перевірка, якої бракує (журнал, етап 27): чи працює правило **вище**
заміряної смуги — сітка 24×24 (1105 труб) за бюджету близько мільйона
симуляцій. Один прогін там — години, тому запускати його варто там, де
машина швидша за контейнер.

    ROUND_SHARE=off NET=Grid24x24 SEED=1 SIMS=1000000 \\
        python experiments/round_rule_ab.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver.island import beam  # noqa: E402
from pipeopt.solver.island.state import IslandState  # noqa: E402

SHARE = os.environ.get("ROUND_SHARE", "0.04")
COUNT = {"rounds": 0, "narrowed": 0}

if SHARE == "off":
    # Не `ROUND_BUDGET_SHARE = 0`: нульова частка дала б нульову межу,
    # тобто звуження на кожному раунді — рівно навпаки. Вимикає правило
    # нескінченна межа.
    def no_limit(self, time_budget):
        """Підміна мусить повторювати сигнатуру справжнього методу."""
        return float("inf")

    IslandState.round_cost_limit = no_limit
else:
    IslandState.ROUND_BUDGET_SHARE = float(SHARE)

_narrow = IslandState.narrow_round
_beam = beam.BeamSearch._beam_search_and_update


def narrow(self, spent):
    result = _narrow(self, spent)
    COUNT["narrowed"] += bool(result)
    return result


def beam_counted(self, *args, **kwargs):
    COUNT["rounds"] += 1
    return _beam(self, *args, **kwargs)


IslandState.narrow_round = narrow
beam.BeamSearch._beam_search_and_update = beam_counted

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
print(f">>> ROUND_SHARE={SHARE} раундів={COUNT['rounds']} "
      f"звужень={COUNT['narrowed']}")
