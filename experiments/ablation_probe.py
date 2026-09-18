"""Ablation: вимкнути один механізм і порівняти за однакового бюджету.

Перепис (`mechanism_census.py`) відповідає на питання «чи механізм
виконується і скільки коштує». Він **не** відповідає на головне: чи від
нього стає краще. Механізм може коштувати 4 % бюджету й тримати весь
результат; може коштувати стільки ж і не давати нічого.

Тут вимикається рівно один механізм, бюджет симуляцій лишається тим
самим, і порівнюється підсумкова вартість. Вимикання зроблене
**підміною методів**, а не правкою продукційного коду: стенд не має
права змінювати те, що міряє.

    ABLATE=swap_search NET=Hanoi SEED=42 SIMS=150000 \\
        python experiments/ablation_probe.py

`ABLATE=none` — контроль. При ньому результат мусить збігтися з
контрольним прогоном побітово; це перше, що варто перевірити на новому
механізмі, бо підміна, яка сама зсуває RNG, знеціднює весь замір.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver import kick_memory, pool, seeds  # noqa: E402
from pipeopt.solver.island import beam, diversification, exchange  # noqa: E402
from pipeopt.solver.local_search import LocalSearch  # noqa: E402
from pipeopt.solver.strategy_selector import StrategySelector  # noqa: E402

WHAT = os.environ.get("ABLATE", "none")


def off_swap_search():
    """Пошук обмінів: повертає вхід без жодної симуляції."""
    LocalSearch.swap_search = lambda self, indices, dyn_bonus: list(indices)


def off_kicks():
    """Увесь шар диверсифікації: кіки не застосовуються взагалі."""
    diversification.Diversification._apply_kick = lambda self, *a, **k: None


def off_bandit():
    """UCB1: вибір стратегії стає рівномірним (VND)."""
    original = StrategySelector.pick

    def pick(self, pool_, temperature, effective_temperature, is_late_game):
        self.ablation = True
        return original(self, pool_, temperature, effective_temperature, is_late_game)

    StrategySelector.pick = pick


def off_kick_memory():
    """Пам'ять невдалих кіків: стратегії щоразу бачать чисту дошку."""
    kick_memory.KickMemory.failed_pipes_for = lambda self, strategy: {}
    kick_memory.KickMemory.as_kick_args = lambda self: {}


def off_consensus_locks():
    """Замки за консенсусом архіву: глибокий спуск нічого не береже."""
    diversification.Diversification._consensus_locks = lambda self, *a, **k: set()


def off_mini_restart():
    """Мінірестарт: свіжі зерна в пул не підкидаються."""
    beam.BeamSearch._check_mini_restart = lambda self, *a, **k: None


def off_emergency_respawn():
    """Аварійне відновлення після тупика променя."""
    beam.BeamSearch._emergency_respawn = lambda self, *a, **k: None


def off_ghost_shield():
    """Щит від ілюзій: підозріло великий виграш більше не перевіряється."""
    from pipeopt.solver.island.state import IslandState
    IslandState._is_ghost_gain = lambda self, cost, solution: False


def off_tabu():
    """Табу на вже бачені рішення в промені."""
    pool.SolutionPool.is_tabu = lambda self, sol, cost: False


def off_beam_deep_polish():
    """Глибока поліровка переможців раунду (5 і 3 проходи)."""
    original = LocalSearch.gradient_squeeze

    def squeeze(self, indices, *args, **kwargs):
        if kwargs.get("max_passes") in (3, 5) and "locked_pipes" not in kwargs:
            return list(indices)
        return original(self, indices, *args, **kwargs)

    LocalSearch.gradient_squeeze = squeeze


def off_reserve_pool():
    """Резервний пул випадкових вилікуваних рішень."""
    seeds.SeedFactory.make_reserve_pool = lambda self, size=4: []


def off_spatial_crossover():
    """Просторове схрещування при міграції."""
    exchange.IslandExchange._spatial_crossover = lambda self, *a, **k: None


SWITCHES = {"none": lambda: None,
            "swap_search": off_swap_search,
            "kicks": off_kicks,
            "bandit": off_bandit,
            "kick_memory": off_kick_memory,
            "consensus_locks": off_consensus_locks,
            "mini_restart": off_mini_restart,
            "emergency_respawn": off_emergency_respawn,
            "ghost_shield": off_ghost_shield,
            "tabu": off_tabu,
            "beam_deep_polish": off_beam_deep_polish,
            "reserve_pool": off_reserve_pool,
            "spatial_crossover": off_spatial_crossover}

assert WHAT in SWITCHES, (
    f"невідомий механізм {WHAT!r}; доступні: {', '.join(sorted(SWITCHES))}")
SWITCHES[WHAT]()

import main  # noqa: E402

NET = os.environ.get("NET", "Hanoi")
NETS = {"Hanoi": ("Hanoi.inp", "in", "30"),
        "Foss": ("Foss.inp", "mm", "40"),
        "Balerma": ("Balerma.inp", "mm", "20"),
        "Grid16x16": ("Grid16x16.inp", "mm", "20"),
        "Grid24x24": ("Grid24x24.inp", "mm", "20")}
inp, units, hmin = NETS[NET]

argv = ["main.py",
        "--inp", os.path.join(ROOT, "InputData", NET, inp),
        "--costs", os.path.join(ROOT, "InputData", NET, "costs.csv"),
        "--units", units, "--hmin", hmin,
        "--seed", os.environ.get("SEED", "42"),
        "--max_sims", os.environ.get("SIMS", "150000"),
        "--time_limit", "0"]
if os.environ.get("PARALLEL"):
    argv += ["--cores", os.environ.get("CORES", "2")]
else:
    argv += ["--cores", "1", "--sequential"]
sys.argv = argv

main.main()
