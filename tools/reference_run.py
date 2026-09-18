"""Еталонні прогони Ханої: перевірка «поведінка не змінилась».

Навіщо це окремим файлом. Критерій кожного етапу рефакторингу —
«еталон збігається побітово», але доти він існував лише як рядок у
документі, який треба було набирати руками з чотирма різними наборами
прапорців. Раз набрати з помилкою — і етап здається зеленим.

Запуск:

    python tools/reference_run.py                 # прогнати й показати
    python tools/reference_run.py --save before.json
    python tools/reference_run.py --compare before.json

`--compare` — це і є перевірка етапу: він друкує таблицю «до / після» і
завершується кодом 1, якщо щось розійшлось. Порівнюються **вартість і
кількість симуляцій**; час прогону ігнорується, бо він залежить від
машини й ніколи не збігається.

Чому саме ці чотири прогони: два зерна на короткому бюджеті ловлять
ранні гілки пошуку, довший бюджет — пізні (заморозку консенсусу, часту
міграцію), а `fast_analytical` — увесь свіп швидкостей, якого немає в
основному режимі. Разом близько 40 секунд.

`--sequential` обов'язкове: лише цей режим відтворюваний побітово (§12.4
`REFACTORING_PLAN.md`). `--units in` теж: таблиця цін Ханої в дюймах, і
без цього прогін дає тиски порядку −10⁸ (§18.5).

## Чому цього замало: `--islands`

Саме `--sequential` робить еталон відтворюваним — і саме він **вимикає
половину солвера**. У послідовному режимі `shared_progress is None`, тому
не виконуються ані міграція між островами, ані IPC, ані порятунок
відсталого воркера. Тобто зелений еталон нічого не каже про весь
міжострівний шар.

`--islands` закриває цю діру. Паралельний прогін **не** відтворюваний
побітово — острови біжать асинхронно, і порядок подій залежить від
планувальника, — тому тут порівнюються не числа, а **структурні факти**:

* прогін завершився без винятків і жоден воркер не впав;
* кожен острів справді працював (є лог, є симуляції);
* між островами був обмін — міграція спрацювала хоч раз;
* підсумкове рішення допустиме за тиском.

Це не заміна еталона, а друга його половина: еталон каже «числа ті самі»,
`--islands` — «механізми, яких еталон не бачить, живі».
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "InputData" / "Hanoi" / "Hanoi.inp"
COSTS = ROOT / "InputData" / "Hanoi" / "costs.csv"

#: (назва, додаткові прапорці). Назва — ключ у звіті, тому стабільна.
CASES = (
    ("analytical/seed42/40k", ["--seed", "42", "--max_sims", "40000",
                               "--run_mode", "analytical"]),
    ("analytical/seed7/40k", ["--seed", "7", "--max_sims", "40000",
                              "--run_mode", "analytical"]),
    ("analytical/seed42/120k", ["--seed", "42", "--max_sims", "120000",
                                "--run_mode", "analytical"]),
    ("fast/seed42", ["--seed", "42", "--run_mode", "fast_analytical"]),
)

#: Де лежить локальний знімок еталона. Він **не** комітиться: числа
#: залежать від машини (див. `environment()`), тому спільного «правильного»
#: знімка не буває — кожен робить свій на своїй машині.
BASELINE = ROOT / "tests" / "reference_baseline.json"

COST = re.compile(r"FINAL RESULT: ([\d.]+)M\$")
SIMS = re.compile(r"Sims: ([\d,]+)")
EMERGENCY = re.compile(r"\[EMERGENCY\]")


def environment():
    """Відбиток середовища, у якому знято знімок.

    Прогін відтворюваний за `--seed` **на тій самій машині й тій самій
    збірці**. Версія Python, платформа і — головне — чи ввімкнений JIT
    numba впливають на порядок операцій із плаваючою комою, тому знімок,
    знятий деінде, порівнювати з тутешнім немає сенсу. Ці поля існують,
    щоб розходження не списали на регресію, коли причина — інша машина.
    """
    try:
        import numba  # noqa: F401
        has_numba = True
    except ImportError:
        has_numba = False
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numba": has_numba and os.environ.get("NUMBA_DISABLE_JIT") != "1",
    }


def count_emergencies():
    """Скільки разів промінь зайшов у тупик.

    Рахується по логах воркерів, а не по stdout: `ctx.log` пише у файл
    прогону, тому в консолі цього рядка немає. Це не критерій збігу, але
    величина, яка багато пояснює при розходженні.
    """
    runs = ROOT / "OutputDataExperiments"
    if not runs.exists():
        return 0
    total = 0
    for log in runs.rglob("*.txt"):
        try:
            total += len(EMERGENCY.findall(log.read_text(encoding="utf-8",
                                                         errors="replace")))
        except OSError:
            continue
    return total


def run_case(flags, timeout):
    # `PYTHONIOENCODING` — не косметика. `capture_output` робить stdout
    # дочірнього процесу трубою, і Python бере для неї кодування локалі:
    # на Windows це cp1251, у якому емодзі з логів солвера не існує. Без
    # цього рядка прогін падає на першому ж рядку виводу (15.09.2026).
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0",
               PYTHONIOENCODING="utf-8")
    command = [sys.executable, str(ROOT / "main.py"),
               "--inp", str(INP), "--costs", str(COSTS),
               "--units", "in", "--cores", "1", "--sequential", *flags]
    done = subprocess.run(command, cwd=ROOT, env=env, timeout=timeout,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    out = done.stdout + done.stderr
    cost = COST.findall(out)
    sims = SIMS.findall(out)
    if not cost:
        raise RuntimeError(f"прогін не дав результату:\n{out[-2000:]}")
    return {
        "cost": float(cost[-1]),
        "sims": int(sims[-1].replace(",", "")) if sims else None,
        "emergency": count_emergencies(),
    }


def collect(timeout):
    results = {}
    for name, flags in CASES:
        print(f"  {name} ...", end="", flush=True)
        # Каталог прогонів чиститься перед кожним випадком, щоб лічильник
        # тупиків стосувався саме цього прогону, а не всіх попередніх.
        shutil.rmtree(ROOT / "OutputDataExperiments", ignore_errors=True)
        results[name] = run_case(flags, timeout)
        r = results[name]
        print(f" {r['cost']:.4f}M$  sims={r['sims']:,}  emergency={r['emergency']}")
    shutil.rmtree(ROOT / "OutputDataExperiments", ignore_errors=True)
    return {"environment": environment(), "runs": results}


def compare(before, after):
    """True, якщо вартість і кількість симуляцій збіглися в усіх прогонах.

    Повертає також список зауважень про середовище: розбіжність у ньому
    не робить порівняння недійсним, але пояснює, чому числа могли
    поїхати без жодної зміни в коді.
    """
    before_env = before.get("environment", {})
    after_env = after.get("environment", {})
    for key in ("python", "platform", "numba"):
        if key in before_env and before_env[key] != after_env.get(key):
            print(f"  [УВАГА] середовище відрізняється: {key} "
                  f"{before_env[key]!r} -> {after_env.get(key)!r}")

    before, after = before.get("runs", before), after.get("runs", after)
    print(f"\n{'прогін':<24} {'було':>12} {'стало':>12}  {'симуляції':>21}")
    identical = True
    for name, _ in CASES:
        b, a = before.get(name), after.get(name)
        if b is None:
            print(f"{name:<24} {'—':>12} {a['cost']:>11.4f}M  (нового прогону не було в еталоні)")
            identical = False
            continue
        same = (b["cost"] == a["cost"] and b["sims"] == a["sims"])
        identical &= same
        mark = " " if same else "  <-- РОЗХОДЖЕННЯ"
        print(f"{name:<24} {b['cost']:>11.4f}M {a['cost']:>11.4f}M  "
              f"{b['sims']:>9,} -> {a['sims']:>9,}{mark}")
    return identical


#: Прогін для перевірки островів: короткий, але достатній, щоб міграція
#: встигла спрацювати. Замір на трьох прогонах: 5-9 міграцій за прогін.
ISLAND_CHECK = {"cores": "2", "max_sims": "30000", "time_limit": "45"}

MIGRATION_MARKS = ("[MIGRATION]", "[HYBRID SUCCESS]", "[HYBRID FAILED]",
                   "[EXPLORATION SHIELD]", "[IPC]", "[RESCUE]")


def island_check(timeout=1800, cores=None):
    """Прогін на кількох островах: перевірка міжострівного шару.

    Повертає словник фактів; рішення «пройшло чи ні» ухвалює викликач
    (`main` друкує, тест перевіряє). Числа навмисне **не** повертаються
    як критерій: паралельний прогін не відтворюваний, і порівнювати їх
    означало б отримати тест, який падає через планувальник.
    """
    cores = cores or ISLAND_CHECK["cores"]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    shutil.rmtree(ROOT / "OutputDataExperiments", ignore_errors=True)
    done = subprocess.run(
        [sys.executable, str(ROOT / "main.py"),
         "--inp", str(INP), "--costs", str(COSTS), "--units", "in",
         "--cores", cores, "--max_sims", ISLAND_CHECK["max_sims"],
         "--time_limit", ISLAND_CHECK["time_limit"], "--run_mode", "analytical"],
        cwd=ROOT, env=env, timeout=timeout, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    out = done.stdout + done.stderr

    runs = ROOT / "OutputDataExperiments"
    worker_logs = sorted(runs.rglob("worker_*.txt")) if runs.exists() else []
    logs_text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                          for p in worker_logs)

    facts = {
        "exit_code": done.returncode,
        "crashed": "Traceback" in out or "crashed:" in out,
        "islands_expected": int(cores),
        "islands_with_log": len(worker_logs),
        # Кожен воркер друкує [TIGHTNESS] у кінці `run()` — отже рядок є
        # доказом, що острів дійшов до кінця, а не помер посередині.
        "islands_finished": logs_text.count("[TIGHTNESS]"),
        "cross_island_events": sum(logs_text.count(m) for m in MIGRATION_MARKS),
        "feasible": "| OK |" in out or "| OK" in out,
        "cost": (float(COST.findall(out)[-1]) if COST.findall(out) else None),
    }
    shutil.rmtree(runs, ignore_errors=True)
    return facts


def island_problems(facts):
    """Перелік того, що пішло не так; порожній — усе гаразд."""
    problems = []
    if facts["exit_code"] != 0:
        problems.append(f"прогін завершився кодом {facts['exit_code']}")
    if facts["crashed"]:
        problems.append("у виводі є Traceback або впав воркер")
    if facts["islands_finished"] < facts["islands_expected"]:
        problems.append(
            f"дійшли до кінця {facts['islands_finished']} островів "
            f"із {facts['islands_expected']}")
    if facts["cross_island_events"] == 0:
        problems.append(
            "жодної події між островами — міжострівний шар не виконувався, "
            "і ця перевірка нічого не перевірила")
    if not facts["feasible"]:
        problems.append("підсумкове рішення недопустиме за тиском")
    return problems


def save(path, results):
    """Записати знімок, **не втративши** список підтверджень.

    `confirmed_on` — це не результат прогону, а накопичений факт: на яких
    машинах ці числа вже відтворились. Перезняття знімка стирає числа
    навмисно, а історію підтверджень — ні.
    """
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    confirmed = existing.get("confirmed_on")
    if confirmed and runs_changed(existing, results):
        # Числа інші — старі підтвердження стосувались інших чисел.
        confirmed = None
    if confirmed:
        results = dict(results, confirmed_on=confirmed)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def runs_changed(before, after):
    return before.get("runs") != after.get("runs")


def add_confirmation(path, snapshot, environment):
    """Дописати середовище до списку підтверджень знімка.

    Сенс списку: знімок комітиться в репозиторій, і питання «чи ці числа
    відтворюються деінде» має відповідь **даними**, а не припущенням.
    Кожен рядок — машина, на якій `--compare` зійшовся побітово.
    """
    confirmed = list(snapshot.get("confirmed_on", []))
    if environment in confirmed:
        return False
    confirmed.append(environment)
    snapshot = dict(snapshot, confirmed_on=confirmed)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", metavar="FILE", nargs="?", const=str(BASELINE),
                        help=f"записати знімок у JSON (без значення: {BASELINE.name} "
                             "у tests/ — саме той файл, який читає pytest)")
    parser.add_argument("--compare", metavar="FILE", nargs="?", const=str(BASELINE),
                        help="звірити з раніше збереженим знімком "
                             f"(без значення: {BASELINE.name})")
    parser.add_argument("--confirm", action="store_true",
                        help="при збігу дописати це середовище до списку "
                             "підтверджень у знімку (лише разом з --compare)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="ліміт на один прогін, секунд")
    parser.add_argument("--islands", action="store_true",
                        help="перевірити міжострівний шар (міграція, IPC), "
                             "якого послідовний еталон не бачить")
    args = parser.parse_args()

    if not INP.exists():
        sys.exit(f"немає {INP}: еталон рахується на Ханої")

    if args.islands:
        print(f"Прогін на {ISLAND_CHECK['cores']} островах "
              "(паралельний, числа не порівнюються)...")
        facts = island_check(args.timeout)
        for key, value in facts.items():
            print(f"  {key}: {value}")
        problems = island_problems(facts)
        if problems:
            print("\nПРОБЛЕМИ:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("\nМІЖОСТРІВНИЙ ШАР ЖИВИЙ: усі острови відпрацювали, "
              f"{facts['cross_island_events']} подій обміну, рішення допустиме.")
        return 0

    print("Еталонні прогони (близько 40 с):")
    results = collect(args.timeout)

    if args.save:
        save(Path(args.save), results)
        print(f"\nЗбережено: {args.save}")

    if args.compare:
        path = Path(args.compare)
        before = json.loads(path.read_text(encoding="utf-8"))
        if compare(before, results):
            print("\nЗБІГ ПОБІТОВО — поведінка не змінилась.")
            if args.confirm:
                added = add_confirmation(path, before, results["environment"])
                print("Середовище дописано до списку підтверджень."
                      if added else
                      "Це середовище вже є у списку підтверджень.")
            return 0
        print("\nПОВЕДІНКА ЗМІНИЛАСЬ. Якщо це свідомо — запишіть у CHANGELOG.md,\n"
              "чому саме, і додайте A/B. Якщо ні — це регресія.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
