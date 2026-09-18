"""Перепис механізмів солвера: чи виконується, скільки коштує, чи не вхолосту.

Питання, на яке відповідає стенд, — перше з трьох, і найдешевше. Перш ніж
питати «що механізм дає якості пошуку» (а це ablation-прогін на кожен, по
кілька зерен і мереж), варто спитати простіше:

* **чи він узагалі виконується** на цій мережі;
* **скільки симуляцій** на нього припадає;
* **яка частка викликів холоста** — повернула те саме, що отримала, або
  повідомила невдачу.

Механізм, який не виконується, ablation-заміру не потребує: порівнювати
нема з чим. Механізм, у якого 100 % викликів холості, теж: він уже
відповів. Тому один прогін на мережу відсікає більшість реєстру, і дорогі
заміри лишаються для тих, хто справді працює.

Стенд **нічого не змінює**: кожна обгортка лише рахує й передає виклик
далі. Порядок випадкових чисел не зсувається — обгортки не тягнуть RNG.

    NET=Hanoi SEED=42 SIMS=150000 OUT=census.json \
        python experiments/mechanism_census.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pipeopt.solver import kicks  # noqa: E402
from pipeopt.solver import seeds  # noqa: E402
from pipeopt.solver import strategy_selector  # noqa: E402
from pipeopt.solver.island import beam, diversification, exchange  # noqa: E402
from pipeopt.solver.local_search import LocalSearch  # noqa: E402

OUT = os.environ["OUT"]

#: label -> {calls, sims, own, idle}. `sims` — усе, що витратив виклик
#: разом із вкладеними; `own` — без них. Без цього поділу частки
#: сумуються далеко за 100 %: спуск лежить усередині оцінки кандидата,
#: оцінка — усередині покоління мутацій, і одні й ті самі симуляції
#: рахуються тричі.
#:
#: `idle` рахується лише там, де «холостий» має сенс: кік повідомив
#: невдачу, ремонт не вдався, спуск повернув те саме рішення.
STATS = {}
CTX = {"ref": None}

#: Глибина вкладеності: кожен кадр накопичує симуляції вкладених у нього
#: обгорнутих викликів.
NESTED = []


def note(label, sims, own, idle):
    entry = STATS.setdefault(label, {"calls": 0, "sims": 0, "own": 0,
                                     "idle": 0, "idle_sims": 0})
    entry["calls"] += 1
    entry["sims"] += sims
    entry["own"] += own
    entry["idle"] += bool(idle)
    if idle:
        # Ціна холостих викликів окремо: «40 % викликів нічого не дали» і
        # «на них пішло 40 % бюджету» — різні твердження, і перше без
        # другого нічого не вирішує.
        entry["idle_sims"] += own


def sim_count():
    ctx = CTX["ref"]
    return ctx.sim_count if ctx is not None else 0


def wrap(owner, name, label, idle_check=None):
    """Обгорнути метод лічильником. `idle_check(result, args)` -> bool."""
    original = getattr(owner, name)

    STATS.setdefault(label, {"calls": 0, "sims": 0, "own": 0,
                             "idle": 0, "idle_sims": 0})

    def inner(self, *args, **kwargs):
        if CTX["ref"] is None and hasattr(self, "ctx"):
            CTX["ref"] = self.ctx
        before = sim_count()
        NESTED.append(0)
        try:
            result = original(self, *args, **kwargs)
        finally:
            nested = NESTED.pop()
            total = sim_count() - before
            if NESTED:
                NESTED[-1] += total
        idle = False
        if idle_check is not None:
            try:
                idle = idle_check(result, args)
            except Exception:
                idle = False
        note(label, total, total - nested, idle)
        return result

    setattr(owner, name, inner)


def same_solution(result, args):
    """Спуск повернув те саме, що отримав, — прохід нічого не змінив."""
    return bool(args) and list(result) == list(args[0])


def kick_failed(result, _args):
    return bool(getattr(result, "failed", False))


def returned_nothing(result, _args):
    return result is None or result is False or result == []


# --- насіння ----------------------------------------------------------
for name in ("make_diverse_seeds", "make_reserve_pool", "make_backbone_seed",
             "make_diverse_seeds_for_fast"):
    wrap(seeds.SeedFactory, name, f"seed/{name}",
         lambda r, a: not r)
wrap(seeds.SeedFactory, "calculate_ideal_d", "seed/calculate_ideal_d", None)

# --- локальний пошук --------------------------------------------------
SQUEEZE_SITE = {1: "ranking", 2: "reserve-seed", 3: "beam-rank<3",
                5: "beam-rank0", 12: "diverse-seed"}


def wrap_squeeze():
    original = LocalSearch.gradient_squeeze

    def inner(self, indices, *args, **kwargs):
        if CTX["ref"] is None:
            CTX["ref"] = self.ctx
        site = SQUEEZE_SITE.get(kwargs.get("max_passes"), "kick-polish")
        before = sim_count()
        NESTED.append(0)
        try:
            result = original(self, indices, *args, **kwargs)
        finally:
            nested = NESTED.pop()
            total = sim_count() - before
            if NESTED:
                NESTED[-1] += total
        idle = list(result) == list(indices)
        note("descent/gradient_squeeze", total, total - nested, idle)
        note(f"descent/squeeze:{site}", total, total - nested, idle)
        return result

    LocalSearch.gradient_squeeze = inner


wrap_squeeze()
wrap(LocalSearch, "heal_network", "descent/heal_network",
     lambda r, a: not r[1])
wrap(LocalSearch, "swap_search", "descent/swap_search", same_solution)
wrap(LocalSearch, "evaluate_candidate", "beam/evaluate_candidate",
     lambda r, a: r[2] is None)
wrap(LocalSearch, "get_high_impact_pipes", "descent/high_impact_pipes", None)

# --- промінь ----------------------------------------------------------
for name, idle in (("_generate_mutations", lambda r, a: not r),
                   ("_beam_search_and_update", None),
                   ("_emergency_respawn", None),
                   ("_emergency_pool_diversity", None),
                   ("_check_mini_restart", returned_nothing),
                   ("_apply_swap", None)):
    wrap(beam.BeamSearch, name, f"beam/{name.lstrip('_')}", idle)

# --- диверсифікація ---------------------------------------------------
for name, idle in (("_apply_kick", None),
                   ("_polish_kicked", None),
                   ("_consensus_locks", lambda r, a: not r),
                   ("_accept_record", None),
                   ("_accept_escape", None),
                   ("_accept_into_pool", None),
                   ("_accept_hail_mary", None),
                   ("_reject", None),
                   ("_note_kick_failure", None),
                   ("_pick_kick_target", None)):
    wrap(diversification.Diversification, name, f"kick/{name.lstrip('_')}", idle)

# --- кожна стратегія окремо -------------------------------------------
STRATEGY_METHODS = [n for n in dir(kicks.KickStrategies)
                    if n.endswith("_kick") or n == "basin_escape"]
assert len(STRATEGY_METHODS) >= 10, (
    f"знайдено лише {len(STRATEGY_METHODS)} стратегій — перевірка дивиться "
    "не туди, і перепис мовчки пропустив би більшість")
for name in STRATEGY_METHODS:
    wrap(kicks.KickStrategies, name, f"strategy/{name}", kick_failed)

# --- вибір стратегії --------------------------------------------------
wrap(strategy_selector.StrategySelector, "pick", "select/pick",
     lambda r, a: not r.bandit_active)   # «холостий» = бандит не працював

# --- міжострівний шар -------------------------------------------------
for name, idle in (("_maybe_migrate", None),
                   ("_migrate_soft", None),
                   ("_migrate_hybrid", None),
                   ("_spatial_crossover", lambda r, a: r is None),
                   ("_process_ipc", None),
                   ("_check_rescue", None),
                   ("_pick_peer_solution", lambda r, a: r is None)):
    wrap(exchange.IslandExchange, name, f"island/{name.lstrip('_')}", idle)


# --- паралельний режим: кожен острів зливає власний перепис ----------
# Воркери — окремі процеси, тому їхні лічильники в батька не повертаються.
# Без цього весь міжострівний шар лишився б із нулями, і перепис сказав би
# «не виконується» там, де він просто відбувається не тут.
#
# Підміняти `run_island_worker` **не можна**: пул передає цю функцію в
# процес через pickle за іменем модуля, і підміна ламає прогін
# (`Can't pickle ...: it's not the same object`). Перша версія стенда так
# і зробила — обидва острови впали, а прогін виглядав успішним, бо
# оркестратор аварію воркера проковтує. Тому обгортається **метод класу**:
# при `fork` дочірній процес успадковує вже підмінений клас.
from pipeopt.solver.island.loop import IslandWorker as _IslandWorker  # noqa: E402

_worker_run = _IslandWorker.run


def _run_counted(self, *args, **kwargs):
    if CTX["ref"] is None:
        CTX["ref"] = self.ctx
    try:
        return _worker_run(self, *args, **kwargs)
    finally:
        dump(suffix=f".w{self.worker_id}")


_IslandWorker.run = _run_counted


def dump(suffix=""):
    ctx = CTX["ref"]
    with open(OUT + suffix, "w") as fh:
        json.dump({"net": os.environ.get("NET", "Hanoi"),
                   "pipes": getattr(ctx, "num_pipes", 0),
                   "total_sims": sim_count(),
                   "mechanisms": STATS}, fh, indent=1)


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
dump()
