# Ревізія коду ПАК підтримки прийняття рішень (GAPipeDiameterSearch)

Дата аналізу: 13.09.2026. Проаналізовано 16 Python-файлів робочого коду (6 531 рядок), без папки `Old/`.
Інструменти: AST-аналіз, `ruff` (pyflakes/ARG/ERA), `radon` (цикломатична складність, MI), порівняння дублікатів через `difflib`.

---

## 0. Стан виконання

| Етап | Стан | Примітка |
|---|---|---|
| 0 — еталонні прогони | **за вами** | потрібно перед етапом 2, не перед 1 |
| 1 — видалення мертвого коду | **зроблено 13.09** | 10 файлів правлено й записано; видалення файлів/папок — за вами (команди в §9) |
| 2 — три дефекти | наступний | змінює результати → спершу еталон |
| 3–12 | попереду | |

**Етап 1, фактично зроблено:** `ruff` з 42 попереджень → 1 (той, що лишився, — `**kwargs` у `forcing_hand_kick`, знімається на етапі 3 разом із `KickResult`). Перевірено прогонами Hanoi у обох режимах: `fast_analytical` → 6.7588 M$, `analytical` (3 воркери, 92 k симуляцій) → 6.0812 M$; у логах воркерів відпрацювали всі змінені шляхи — 231 кік, 33 міграції, 10 POOL FLUSHED (BASIN_ESCAPE), 10 FORCE (через `is_ghost_solution` зі зміненим підписом), 5 SWAP, жодної помилки.

Зміни у 10 файлах (±198 рядків diff):

* `ga_config.py` — прибрано непрочитувані поля `inp_file`, `pop_size`, `n_gens`, `runs`, `mutation_start/end`, `epsilon_start/end`, `init_method`, `analytical_mutation_rate` та методи `get_max_diameter/get_min_diameter`; дефолт `run_mode` змінено з `"ga"` на `"analytical"`; 31 → 21 рядок
* `main.py`, `gui.py` — виклики `GAConfig(...)` без заглушок `pop_size=200, n_gens=150`
* `gui.py` — прибрано `from Old.ga_optimizer import GeneticOptimizer` (останнє посилання живого коду на `Old/`), невикористаний `format_time`, `worker_eval_task`, патч `WaterSimulator.worker_eval_wrapper`, недосяжну гілку `if self.opt_mode.get() == "ga"` у `stop_thread`
* `water_sim.py` — прибрано `death_penalty`
* `plot.py` — прибрано повторний блок імпортів у середині файлу (рядки 116–119) і невикористаний `numpy`
* `analytical_solver/context.py` — прибрано мертвий `baseline_bonus` і невикористаний параметр `cost` у `is_ghost_solution` (з правкою 3 викликів)
* `analytical_solver/orchestrator.py` — прибрано `corridor_pool_streak`, `_last_global_improvement_sim`, `self.EPOCHS`, метод `_build_diverse_archive` (32 рядки), невикористаний параметр `gb` у `_apply_kick`, невикористані параметри `max_sims`/`time_limit_sec` у `solve_standalone`
* `analytical_solver/local_search.py` — закомментований блок «ВСТАВКА ДЛЯ БАЗОВОГО ТЕСТУВАННЯ» став параметром `gradient_squeeze(..., classical_descent=False)`: поведінка за замовчуванням не змінилась, але ablation-режим більше не вимагає правки вихідника
* `analytical_solver/kicks.py` — прибрано невикористані `import itertools` і `cost_limit`
* `analytical_solver/__init__.py` — доданий `__all__`

Не робилось свідомо (лишено на пізніші етапи): `**kwargs` у кіках (етап 3), ініціалізація полів GUI у `__init__` (етап 2, бо змінює поведінку кнопки «ЗУПИНИТИ»), перейменування `ga_*` файлів (етап 11).

---

## 1. Метрики: де саме проблема

| Файл | Рядків | MI | Найгірший метод | CC |
|---|---|---|---|---|
| `analytical_solver/orchestrator.py` | 1 606 | **C (0.00)** | `IslandWorker._apply_kick` | **F (181)** |
| `analytical_solver.py` (корінь) | 1 560 | **C (0.00)** | — | **недосяжний** |
| `gui.py` | 1 330 | **C (0.00)** | `load_viewer_data` | **F (74)** |
| `analytical_solver/kicks.py` | 528 | C (0.00) | `upstream_bottleneck_kick` | D (25) |
| `plot.py` | 318 | A (20.6) | `plot_network_map` | D (26) |
| `main.py` | 317 | A (21.9) | `main` | D (30) |
| `analytical_solver/local_search.py` | 249 | A (25.7) | `gradient_squeeze` | E (32) |
| `water_sim.py` | 205 | A (26.2) | `__init__` | B |
| решта (context, pool, cache, fast_math, ga_*) | 218 | A | — | ≤ C |

Три файли з індексом підтримуваності **0.00** тримають 68 % коду. Решта модулів — у нормі; проблема локалізована, і це добра новина: рефакторинг можна робити точково.

Методи, що перевищують розумну межу (≈ 50 рядків / CC ≈ 15):

| Метод | Рядків | CC |
|---|---|---|
| `IslandWorker._apply_kick` | 350 | 181 |
| `AnalyticalSolver.solve_standalone` | 207 | 57 |
| `IslandWorker.run` | 191 | 46 |
| `PipelineOptimizerApp.load_viewer_data` | 243 | 74 |
| `PipelineOptimizerApp.run_optimization` | 171 | 34 |
| `main()` | 154 | 30 |
| `_update_live_network_graph` | 89 | 33 |
| `SeedFactory.make_warm_seeds` | 95 | 37 |
| `_beam_search_and_update` | 79 | 26 |
| `_setup_ui` | 131 | B |

---

## 2. Мертвий код (підтверджено, не припущення)

### 2.1 `analytical_solver.py` у корені — 1 560 рядків, які ніколи не виконуються

Поряд існують модуль `analytical_solver.py` і пакет `analytical_solver/`. **Python завжди обирає пакет**: у `FileFinder` обробники каталогів перевіряються раніше за файлові. Перевірено експериментально:

```
pkg.py        → X = "MODULE_FILE"
pkg/__init__.py → X = "PACKAGE_DIR"
import pkg    → PACKAGE_DIR
```

Отже `from analytical_solver import AnalyticalSolver` у `main.py:16` та `gui.py:23` завантажує пакет. Файл у корені — це попередня генерація коду, що вже розійшлася з пакетом:

| Клас | Монолiт | Пакет | Схожість |
|---|---|---|---|
| `LRUCache` | 16 | 16 | 1.00 |
| `SolverContext` | 132 | 158 | 0.77 |
| `SolutionPool` | 46 | 51 | 0.40 |
| `LocalSearch` | 197 | 245 | 0.45 |
| `KickStrategies` | 392 | 524 | 0.26 |
| `AnalyticalSolver` | 745 | 383 | 0.19 |

**Дія:** видалити `analytical_solver.py` (історія є в git). Це −24 % коду проєкту одним комітом і нульовий ризик. Саме він створює найгірший ефект «код виглядає масивним»: при пошуку по проєкту кожен метод знаходиться двічі у двох різних версіях.

### 2.2 Мавпячий патч, що прихововує дублікат

```python
# gui.py:1329, main.py:238
AnalyticalSolver.worker_task = staticmethod(analytical_worker_task)
```

`AnalyticalSolver.worker_task` (orchestrator.py:1227–1272, 46 рядків) перезаписується під час старту. Тобто в репозиторії живуть **три** майже однакові реалізації запуску воркера:

| Розташування | Рядків | Схожість |
|---|---|---|
| `orchestrator.py:1227` `worker_task` (classmethod) | 46 | базова |
| `main.py:79` `analytical_worker_task` | 82 | 1.00 з gui.py |
| `gui.py:108` `analytical_worker_task` | 82 | 1.00 з main.py |

Дві останні — **побайтово ідентичні** (`difflib` similarity = 1.00), включно з вкладеним класом `WorkerLogger`. До того ж вони дістають класи через приватну кухню імпорту:

```python
solver_module = sys.modules[AnalyticalSolver.__module__]
SolverContext = solver_module.SolverContext   # замість from .context import SolverContext
```

Це працює лише тому, що `orchestrator.py` випадково має ці імена у своєму просторі імен. Будь-яка зміна імпортів у пакеті зламає GUI без жодної помилки на етапі імпорту.

### 2.3 Машинерія епох — підтверджений історичний артефакт

**Уточнено автором:** система епох справді використовувалась на початку, але була замінена однією епохою з острівною моделлю та асинхронною синхронізацією. Отже це не «випадково відключена функція», а залишок попередньої архітектури — видаляється, не відновлюється.

Це узгоджується з кодом: роль, яку в багатоепохному режимі виконували warm seeds (перезапуск воркерів з глобального архіву на початку кожної епохи), тепер виконує асинхронна синхронізація всередині одної епохи — `_process_ipc` (пасивна інжекція рішень сусідів), `_check_rescue` (прийняття global best при відставанні), міграція зі `_spatial_crossover` (рядки 399–459) та `_check_mini_restart` (м'який перезапуск з архіву). Тобто механізм не зник, він переїхав у цикл раунду.



```python
# orchestrator.py:1410
epochs = 1
for epoch in range(epochs):        # завжди рівно одна ітерація, epoch == 0
```

```python
# orchestrator.py:299
seeds = self.seeder.make_diverse_seeds() if self.epoch == 0 else self.seeder.make_warm_seeds(...)
```

Оскільки `epoch` завжди 0, недосяжними є:

* `SeedFactory.make_warm_seeds` — 95 рядків, CC 37 (уся система каст EXPLOITER / RELINKER / ARCHITECT / EXPLORER);
* `SeedFactory.greedy_path_relink` — 23 рядки, викликається лише з `make_warm_seeds`;
* параметр `failed_basins` у `make_warm_seeds` (ніде не використовується навіть у тілі);
* `AnalyticalSolver._build_diverse_archive` — 31 рядок, не викликається взагалі нізвідки;
* змінна `global_failed_basins` (рядок 1428) — наповнюється, ніколи не читається;
* `worker_basins` з результату воркера у паралельній гілці — розпаковується і відкидається.

**Що саме видаляти** (етап 9 плану):

| Об'єкт | Місце | Дія |
|---|---|---|
| `SeedFactory.make_warm_seeds` | orchestrator.py:114-208 (95 рядків) | → `attic/` |
| `SeedFactory.greedy_path_relink` | orchestrator.py:210-232 (23) | → `attic/` (викликається лише з `make_warm_seeds`) |
| тернарник вибору seeds | orchestrator.py:299 | → `seeds = self.seeder.make_diverse_seeds()` |
| `IslandWorker.__init__(..., epoch)` + `self.epoch` | 235, 245 | видалити параметр (єдине читання — рядок 299) |
| `IslandWorker.__init__(..., global_archive)` | 235, 244 | **видалити параметр**, лишити `self.global_archive = []`: у `solve_standalone:1406` це завжди `[]`, бо епоха одна, а реальне значення приходить із `shared_progress` у рядку 403 |
| `self.EPOCHS = {...}` | 1292 | мертвий словник (присвоюється, не читається) |
| `epochs = 1` + `for epoch in range(epochs)` | 1410, 1431 | розгорнути тіло на один рівень вліво (−1 рівень вкладеності у 130 рядках `solve_standalone`) |
| `global_failed_basins`, `worker_basins` | 1428, 1554-1556 | наповнюється, не читається |
| `AnalyticalSolver._build_diverse_archive` | 1305-1335 (31) | не викликається нізвідки |
| `epoch_start_sims` → `start_sims`, `epoch_sims` → `run_sims` | 310, 358 | перейменувати |
| `cumulative_epoch_sims` → `total_worker_sims` | 1427 + 5 місць | перейменувати |
| `time_per_epoch` → `time_limit` | 1411, 1434, 1441 | перейменувати |
| `epoch` у `WorkerTask` + `STARTING EPOCH {epoch+1}` у логах | main.py:89,115; gui.py:118,144; orchestrator.py:1252 | лог-файли стають `worker_NN.txt` без префікса епохи |

У цьому мертвому коді вже лежить дефект, який ніколи не спрацював саме тому, що код мертвий — додатковий аргумент видаляти, а не «залишити на всяк випадок»:

```python
# make_warm_seeds, orchestrator.py:160-170
if target_sol is None:
    div_seeds = self.make_diverse_seeds()
    ...
else:
    diff_indices = [i for i in range(...) if base_sol[i] != target_sol[i]]

if not diff_indices:        # ← NameError, якщо target_sol is None
```

Виграш: −150 рядків, `solve_standalone` втрачає один рівень вкладеності на 130 рядках, `SeedFactory` скорочується з 212 до ~95 рядків і стає читабельним.

### 2.3.1 Побічний наслідок: діверсифікація островів тепер слабша, ніж передбачав старий дизайн

Це не дефект, а спостереження, що варте вашої уваги як дослідника острівних моделей.

Після переходу на одну епоху **всі воркери стартують з `make_diverse_seeds()`** (orchestrator.py:47-64), а вона детермінована за мережею: три фіксовані швидкості `[1.0, 1.2, 0.8]` → ітерація ідеальних діаметрів → `heal_network` → `gradient_squeeze`. Тобто структурно початкові точки всіх островів однакові; розходяться вони лише через:

* `random.shuffle(active_indices)` у `gradient_squeeze` (порядок обходу труб, різний seed на воркер);
* `make_reserve_pool(size=4)` — справді випадкові резервні рішення;
* `random.seed(seed_mod + i)` у воркері.

У багатоепохному режимі структурну різницю між островами давали саме касти (`make_warm_seeds`: EXPLOITER — мікромутації, EXPLORER — макромутації, ARCHITECT — консенсус, RELINKER — path-relinking, плюс примусовий ADVENTURER). Зараз від цієї диференціації лишився один рудимент — у `_process_ipc:534`:

```python
is_adventurer = (self.n_workers >= 4 and self.worker_id == self.n_workers - 1)
```

тобто останній воркер лише отримує імунітет до інжекцій до `progress_ratio < 0.85`, але стартує так само, як усі інші.

Ймовірно саме тому в коді так багато аварійної діверсифікації: `_emergency_pool_diversity` (резервний пул при падінні середньої відстані Хеммінга), `_emergency_respawn`, `reserve_pool`, `min_dist` у beam. Вони компенсують те, чого немає на старті.

**Дешеве покращення без нового коду.** `make_diverse_seeds_for_fast` (orchestrator.py:66-87) уже робить свіп 16 швидкостей `0.5…2.0` і використовується лише в `solve_fast`. Достатньо розподілити цей свіп між островами:

```python
# seeds.py
def make_island_seeds(self, worker_id: int, n_workers: int) -> list:
    velocities = [v / 10.0 for v in range(5, 21)]      # 0.5 … 2.0
    mine = velocities[worker_id::n_workers]            # воркер 0: 0.5, 0.9, 1.3, 1.7 …
    return [s for v in mine if (s := self._seed_for_velocity(v)) is not None]
```

При 5 островах кожен отримує 3–4 структурно різні початкові точки замість трьох однакових. Це зберігає ідею каст (різні режими роботи островів) без машинерії епох, і — що важливо для дисертації — дає чисту ablation-вісь «однорідний старт vs розподілений свіп», яку можна виміряти на Hanoi і Balerma.

### 2.3.2 Асиметрія послідовного та паралельного режимів

Оскільку `self.global_archive` наповнюється лише з `shared_progress` (рядок 403), а `shared_progress is None` при `self.mp_pool is None` (`solve_standalone:1421-1422`), у **послідовному режимі архів залишається порожнім назавжди**. Отже там ніколи не спрацьовують:

* `BASIN_ESCAPE` — вимагає `len(self.global_archive) >= 2` (рядок 662);
* consensus-locking у `gradient_squeeze` — вимагає `>= 3` (рядок 838);
* міграція та `_check_rescue` — узагалі під `if shared_progress is not None`.

Тобто послідовний прогін — це не «той самий алгоритм на одному ядрі», а суттєво інший алгоритм. Якщо ви колись використовували послідовний режим для відлагодження або як baseline в ablation, ці результати не порівнянні з паралельними. Варто або задокументувати це явно, або дати `solve_standalone` шлях з локальним `dict` замість `manager.dict()` при одному воркері — тоді послідовний режим стає справжнім окремим випадком паралельного.

### 2.4 Залишки режиму GA

Режим GA прибраний з UI (`gui.py:280` пропонує лише «Швидкий Аналітичний» / «Повний Аналітичний»), але його інфраструктура лишилася:

| Залишок | Місце | Стан |
|---|---|---|
| `from Old.ga_optimizer import GeneticOptimizer` | `gui.py:24` | імпорт не використовується; єдине посилання живого коду на `Old/` |
| `worker_eval_task` | `gui.py:97-106` | 10 рядків, не викликається |
| `WaterSimulator.worker_eval_wrapper = ...` | `gui.py:1328` | атрибут не читається ніде в проєкті |
| `if self.opt_mode.get() == "ga"` | `gui.py:440` | недосяжна гілка — значення «ga» неможливе |
| `GAConfig.mutation_start/_end`, `epsilon_start/_end`, `analytical_mutation_rate` | `ga_config.py:13-21` | оголошені, ніде не читаються |
| `GAConfig.get_max_diameter()`, `get_min_diameter()` | `ga_config.py:28-32` | не викликаються |
| `GAConfig.init_method` | `ga_config.py:19` | записується у двох місцях, не читається |
| `GAConfig.run_mode = "ga"` (default) | `ga_config.py:18` | дефолт неможливого режиму |
| `pop_size=200, n_gens=150` | `main.py:209`, `gui.py:504` | прямо підписані як заглушки: `# Стандартні заглушки, щоб не ламався клас` |
| `WaterSimulator.death_penalty` | `water_sim.py:73` | присвоюється, не читається |
| `WaterSimulator.evaluate()` | `water_sim.py:126` | лишився лише як pre-flight перевірка `main.py:220`; параметри `penalty_factor`/`epsilon` — спадок GA |

Клас називається `GAConfig`, файли — `ga_config.py`, `ga_data.py`, `ga_utils.py`, хоча GA в проєкті більше немає. Це найдешевша для виправлення й найпомітніша для читача невідповідність назв.

### 2.5 Стан, що пишеться і не читається

| Атрибут | Записів | Читань |
|---|---|---|
| `IslandWorker.corridor_pool_streak` | 3 (рядки 283, 909, 941) | 0 |
| `IslandWorker._last_global_improvement_sim` | 4 (285, 311, 525, 1218) | 0 |
| `SolverContext.baseline_bonus` | 1 (context.py:62) | 0 — при цьому описаний у `readme.md` §4 |
| `SolutionPool.add_basin_to_tabu()` | визначений | 0 викликів у пакеті |

Останній пункт — не косметика, а дефект. Метод, що додає *підпис басейну*, не використовується; замість нього:

```python
# orchestrator.py:821
if self.run_best_sol: self.pool.basin_tabu.append(tuple(self.run_best_sol))
```

Тут у `basin_tabu` кладеться повний вектор рішення, а перевірка шукає інше:

```python
# pool.py:33
def is_basin_tabu(self, indices):
    return self.get_basin_signature(indices) in self.basin_tabu
```

Для `num_pipes > 50` `get_basin_signature` повертає кортеж із ~25 усереднених чанків, який ніколи не зійдеться з повним вектором. Тобто **basin tabu на Balerma (454 труби) не працює взагалі**, а на Hanoi (34 труби) працює, бо там `get_basin_signature` повертає `tuple(indices)`. Це прямо впливає на інтерпретацію ablation-результатів між двома мережами. Виправлення — один рядок: `self.pool.add_basin_to_tabu(self.run_best_sol)`.

### 2.6 Дрібне, від `ruff`

* `analytical_solver/kicks.py:4` — `import itertools` не використовується
* `analytical_solver/local_search.py:3` — `import numpy as np` не використовується
* `plot.py:5` — `import numpy as np` не використовується
* `plot.py:117-120` — повторний `import os, pandas, matplotlib, FuncFormatter` усередині файлу (F811, вже імпортовані в рядках 3–10)
* `analytical_solver/orchestrator.py:1242` — `import sys` не використовується
* `ga_config.py:2` — `Any` не використовується
* `gui.py:27` — `format_time` імпортований, не використовується
* `local_search.py:38-41` — закомментований блок «ВСТАВКА ДЛЯ БАЗОВОГО ТЕСТУВАННЯ»; це режим ablation, керований правкою вихідника → має стати параметром (див. §4.5)
* невикористані локальні змінні: `up_limit` (local_search.py:186), `cost_limit` (kicks.py:465), `e` в `except ... as e` без використання (context.py:153, gui.py:780)
* невикористані параметри: `cost` у `is_ghost_solution` (context.py:67), `gb` у `_apply_kick` (orchestrator.py:647), `max_sims` і `time_limit_sec` у `solve_standalone` (orchestrator.py:1401 — приймаються й ігноруються, бо беруться з `self`), `mode` у `on_mode_change` (gui.py:424)
* 8 × `f""` без підстановок

### 2.7 Файли у робочій теці

* `build/` — не менше 162 МБ, `dist/` — не менше 130 МБ (лістинг обрізався на 2 000 записів). У `.gitignore` є, але ≈ 292 МБ лежать у робочій теці.
* `temp.bin`, `temp.inp`, `temp.rpt`, `_temp_sim_data/` — сміття від перерваних запусків у корені.
* `text.txt` (8.8 КБ) — без призначення.
* `Hanoi_1M.json`, `balerma_15M.json` — конфіги запусків у корені; місце — `configs/`.
* `Old/` — 8 файлів, 193 КБ; єдине посилання з живого коду — невикористаний імпорт у `gui.py:24`.
* **`tests/` — це не тести.** 22 папки (`Тест_1`…`Тест_4`, `u_test1`…`u_test12`, `best_adaptive`, `best_static`, …) з архівами експериментів: `results_adaptive.csv`, `solution_static_map.png` тощо. Автоматизованих тестів у проєкті **нуль**, а ім'я `tests/`, яке будь-який Python-розробник (і `pytest`) читає як каталог тестів, зайняте результатами. Перейменувати на `experiments/archive/` і звільнити `tests/`.

---

## 3. Дублювання

### 3.1 Запуск оптимізації: `main.py` ↔ `gui.py`

| Функція | `main.py` | `gui.py` | Схожість |
|---|---|---|---|
| `analytical_worker_task` | 82 рядки | 82 рядки | **1.00** |
| `get_temp_root` | 2 | 2 | **1.00** |
| `worker_init` | 25 | 19 | 0.90 |
| `WorkerLogger.writelines` | 3 | 3 | **1.00** |
| `clean_all_temp` | 9 | 37 | 0.17 (розійшлися) |

Крім того `main()` (154 рядки) і `run_optimization()` (171 рядок) повторюють один сценарій: збірка `GAConfig` → `load_config` → `WaterSimulator` → `multiprocessing.Pool` → цикл `runs` → зворотне відображення діаметрів у індекси → `sim.get_stats` → `export_solution` / `plot_network_map` / `plot_convergence` → `runs_summary.csv`. Блок «діаметри → індекси» (`main.py:256-260`, `gui.py:545-549`) ідентичний посимвольно.

### 3.2 Малювання мережі — чотири копії

Блок «унікальні діаметри → `turbo` cmap → `color_indices` → `widths` за формулою `1.0 + 4.0*(d-min)/(max-min+1e-6)` → colorbar із `set_yticklabels`» присутній у:

| Місце | Рядки |
|---|---|
| `plot.py:256-298` (`plot_network_map`, у файл) | ~40 |
| `gui.py:742-774` (`_update_live_network_graph`, live) | ~35 |
| `gui.py:913-931` (`_draw_basic_topology`, прев'ю INP) | ~20 |
| `gui.py:1168-1202` (`load_viewer_data`, вікно результатів) | ~35 |

Чотири копії вже розійшлися у деталях: `plot.py` і `load_viewer_data` малюють баки трикутником, `_update_live_network_graph` і `_draw_basic_topology` — ні; `plot.py` додає легенду, інші — ні. Пошук колонки діаметра (`'diam' in col.lower() or 'діаметр' in col.lower()`) продубльований тричі (`gui.py:871`, `1094`, `1150`).

### 3.3 Повтори формул у солвері

| Вираз | Входжень |
|---|---|
| `self.ctx.simulator.config.h_min` | **43** |
| `self.ctx.get_cached_stats(...)` | 54 |
| `score = cost - (p_surplus * dyn_bonus)` (у варіаціях) | 14 |
| `c, p, feas, _ = get_cached_stats(x); if feas and p >= h_min:` | 11 |

Чотириланковий шлях `self.ctx.simulator.config.h_min` — це закон Деметри у найчистішому вигляді: солвер знає, що симулятор має конфіг, а конфіг — поле. 43 рази.

### 3.4 Найнебезпечніший повтор: `feas` не є feasibility

```python
# water_sim.py:183
return cost, min_p, max_p, crit_node
```

Скрізь у солвері це розпаковується так:

```python
c, p, feas, _ = self.ctx.get_cached_stats(sol)   # ← 3-й елемент це max_p, не ознака feasibility
if feas and p >= h_min: ...
```

Тобто `feas` — це максимальний тиск у мережі (float), а не bool. У помилковому випадку `get_stats` повертає `-1.0`, що **truthy**. Тому `if feas` майже завжди істинне і перевірка є декорацією. Наслідки зараз замасковані тим, що поруч завжди стоїть `p >= h_min`, але в двох місцях `if feas` стоїть самостійно і є фактично no-op:

* `orchestrator.py:1006` — `parent_p_surplus = (p - h_min) if feas else 0.0`
* `orchestrator.py:1099` — `if feas_min:` перед логікою прийняття в beam

Це рівно той тип дефекту, який породжує «магічні» результати ablation. Виправлення в §4.2.

---

## 4. Що і як розбити

### 4.1 `_apply_kick`: 350 рядків, CC 181 → п'ять об'єктів

Зараз один метод виконує шість незалежних задач. Розділити за задачами:

```
analytical_solver/
├── kick_registry.py     KickSpec, KICK_REGISTRY, KickResult
├── strategy_selector.py StrategySelector  (UCB1 / VND, hot/cold статистика, burst-penalty)
├── acceptance.py        AcceptancePolicy  (explosion threshold, water-level, HAIL MARY, basin escape)
├── mutation_rates.py    AdaptiveRates     (JADE: mu_ruin/perturb/spatial/escape)
└── island_worker.py     IslandWorker._apply_kick  ≈ 40 рядків
```

**Крок 1 — уніфікувати контракт кіків.** Зараз 10 методів `KickStrategies` повертають кортежі різної довжини, і четвертий елемент має різне значення залежно від стратегії:

```python
# orchestrator.py:773-776
if len(res) >= 4:
    if strategy in ["BOTTLENECK", "LOOP_BALANCE"]: failed_pipe_id = res[3]
    elif strategy == "TOPO_INV":                   path_sig = res[3]
    elif strategy in ["RUIN_RECREATE", ...]:       used_pct = res[3]
```

Замінити на один тип:

```python
# kick_registry.py
from dataclasses import dataclass

@dataclass
class KickResult:
    solution: list | None = None
    locked: set | None = None
    message: str = ""
    failed_pipe: int | None = None      # BOTTLENECK, LOOP_BALANCE
    path_signature: tuple | None = None # TOPO_INV
    used_pct: float | None = None       # R&R, SMART_PERTURB, SPATIAL_PERTURB, BASIN_ESCAPE

    @property
    def failed(self) -> bool:
        return self.solution is None or self.locked is None
```

Кожен `*_kick` повертає `KickResult(...)` замість `return None, None, ""` / `return kicked, locked, msg, target_pipe`. Зникає 20-рядковий блок розбору `res[3]` і `if not isinstance(res, tuple) or len(res) < 3`.

**Крок 2 — реєстр замість ланцюга `if/elif`.** Зараз 11 гілок `elif strategy == "..."` (рядки 758–768) плюс три окремі списки, де ті самі імена стратегій перелічені текстом (`self.strategies` у `__init__:250`, `all_tracked:255`, `pool_strats` у трьох гілках `_apply_kick:661-670`). Додавання стратегії вимагає правок у п'яти місцях.

```python
# kick_registry.py
@dataclass(frozen=True)
class KickSpec:
    name: str
    method: str                    # ім'я методу в KickStrategies
    temp_range: tuple[float, float] # за якої T доступна
    needs_cycles: bool = False      # виключити для дерев
    adaptive_rate: str | None = None # яку mu_* вчить

KICK_REGISTRY = {
    "SHOCK":       KickSpec("SHOCK", "forcing_hand_kick", (0.0, 0.90)),
    "BOTTLENECK":  KickSpec("BOTTLENECK", "upstream_bottleneck_kick", (0.0, 0.90)),
    "TOPO_INV":    KickSpec("TOPO_INV", "topological_inversion_kick", (0.0, 1.01)),
    "LOOP_BALANCE":KickSpec("LOOP_BALANCE", "loop_balancing_kick", (0.0, 0.90), needs_cycles=True),
    "ZERO_SUM":    KickSpec("ZERO_SUM", "zero_sum_shift_kick", (0.0, 0.90)),
    "TRIM":        KickSpec("TRIM", "peripheral_trim_kick", (0.0, 0.90)),
    "SMART_PERTURB":  KickSpec("SMART_PERTURB", "smart_perturbation_kick", (0.40, 0.90), adaptive_rate="perturb"),
    "SPATIAL_PERTURB":KickSpec("SPATIAL_PERTURB","spatial_perturb_kick", (0.40, 1.01), adaptive_rate="spatial"),
    "RUIN_RECREATE":  KickSpec("RUIN_RECREATE", "ruin_and_recreate_kick", (0.40, 1.01), adaptive_rate="ruin"),
    "BASIN_ESCAPE":   KickSpec("BASIN_ESCAPE", "basin_escape", (0.90, 1.01), adaptive_rate="escape"),
}
```

Тоді виклик — один рядок, а «які стратегії доступні за температури T» стає запитом до реєстру:

```python
spec = KICK_REGISTRY[strategy]
result = getattr(self.kicker, spec.method)(kick_target, **kick_args)
```

**Крок 3 — вибір стратегії окремо.** Рядки 650–719 (UCB1, VND-фолбек, burst-penalty, consecutive-fails, hot/cold статистики) — це самостійний алгоритм із власним станом і нульовою залежністю від решти `_apply_kick`:

```python
# strategy_selector.py
class StrategySelector:
    def __init__(self, tracked_names, is_tree): ...
    def available(self, temperature, archive_size) -> list[str]: ...
    def select(self, temperature, is_late_game) -> str: ...
    def reward(self, name, improvement_pct): ...
    def penalize(self, name, factor): ...
    def mark_failed(self, name): ...
    def decay_all(self, factor=0.3): ...
```

Це одразу дає те, чого зараз немає: `StrategySelector` тестується юніт-тестами без EPANET.

**Крок 4 — політика прийняття окремо.** Рядки 853–996 — каскад `if/elif/elif/elif/else` (пряме оновлення рекорду → basin escape → water-level accept → HAIL MARY → відхилення). Виділити в `AcceptancePolicy.classify(...) -> Verdict`, де `Verdict` — enum. `IslandWorker` тоді лише застосовує вердикт.

**Крок 5 — `ABLATION_MODE` у конфіг.** `ABLATION_MODE = False` (рядок 648) — локальна константа, яку для ablation-серії правлять у вихіднику, і від неї залежать 8 гілок. Має бути полем `SearchConfig`, що приходить із CLI (`--ablation`). Так само `local_search.py:37-41` — закомментований блок «Classical Greedy Descent» замість параметра.

Очікуваний результат: `_apply_kick` ≈ 40 рядків, CC ≈ 12.

### 4.2 Результат симуляції як іменований тип — виправляє §3.4

```python
# water_sim.py
from typing import NamedTuple

class SimStats(NamedTuple):
    cost: float
    p_min: float
    p_max: float
    crit_node: str

    @property
    def simulation_ok(self) -> bool:
        return self.crit_node != "ERR"

    def is_feasible(self, h_min: float) -> bool:
        return self.simulation_ok and self.p_min >= h_min

    def surplus(self, h_min: float) -> float:
        return self.p_min - h_min
```

`get_stats` повертає `SimStats(cost, min_p, max_p, crit_node)` — кортеж, тому весь наявний код із `c, p, feas, _ = ...` продовжує працювати під час переходу. Далі поступово:

```python
# було
c, p, feas, _ = self.ctx.get_cached_stats(sol)
if feas and p >= self.ctx.simulator.config.h_min:
    score = c - ((p - self.ctx.simulator.config.h_min) * dyn_bonus)

# стало
st = self.ctx.stats(sol)
if st.is_feasible(self.ctx.h_min):
    score = self.ctx.score(st, dyn_bonus)
```

Разом із цим у `SolverContext`:

```python
# context.py
self.h_min = simulator.config.h_min          # прибирає 43 входження simulator.config.h_min

def score(self, st: SimStats, dyn_bonus: float) -> float:
    surplus = st.surplus(self.h_min)
    if surplus < 0:
        return float('inf')
    effective_bonus = dyn_bonus * 0.2 if surplus > 10.0 else dyn_bonus
    return st.cost - surplus * effective_bonus
```

Правило «при надлишку > 10 м бонус ×0.2» зараз повторене в чотирьох місцях (`orchestrator.py:644`, `875`, `1106`; `local_search.py` у трьох варіантах без нього) — і саме тому вже розійшлося. Одна функція `score` робить формулу penalized-score, описану в `readme.md` §2, єдиною у коді.

### 4.3 Стан воркера: 30 атрибутів і 16 `getattr`/`hasattr`

`IslandWorker.__init__` приймає 11 позиційних аргументів і створює ~30 атрибутів; ще частина з'являється пізніше в `run` (`reserve_pool`, `ipc_immunity`, `stag_limit`, `_last_ping_sims`, `_migration_interval_sims`, `_force_flush_next`, `_last_strategy`, `_strategy_burst_count`). Звідси 16 захисних `getattr(self, 'x', default)` / `if not hasattr(self, 'x')` у `_apply_kick` — код захищається від власного конструктора:

```python
# orchestrator.py:736-739
if not hasattr(self, 'bottleneck_failed_pipes'): self.bottleneck_failed_pipes = {}
if not hasattr(self, 'loop_balance_failed_pipes'): self.loop_balance_failed_pipes = {}
if not hasattr(self, 'zero_sum_tabu'): self.zero_sum_tabu = {}
if not hasattr(self, 'bottleneck_boosted_pipes'): self.bottleneck_boosted_pipes = {}
```

(перші три вже є в `__init__:284-287` — ці рядки не роблять нічого; четвертого в `__init__` немає — і ось для нього `hasattr` справді потрібен, що і є доказом проблеми)

Розділити на три dataclass:

```python
@dataclass(frozen=True)
class WorkerConfig:        # незмінне: те, що зараз 11 аргументів
    worker_id: int
    n_workers: int
    max_sims: float
    beam_width: int
    network_class: str
    epoch: int
    ablation: bool = False

@dataclass
class SearchState:         # змінне: прогрес пошуку
    stagnation_counter: int = 0
    run_best_cost: float = float('inf')
    run_best_sol: list | None = None
    global_best_cost: float = float('inf')
    last_published_cost: float = float('inf')
    progress_ratio: float = 0.0
    ipc_immunity: int = 30
    stag_limit: int = 4
    force_flush_next: bool = False

    @property
    def is_late_game(self) -> bool:      # прибирає розсинхрон self.is_late_game / локальної is_late_game
        return self.progress_ratio > 0.5

@dataclass
class KickMemory:          # tabu-пам'ять стратегій
    bottleneck_failed: dict = field(default_factory=dict)
    bottleneck_boosted: dict = field(default_factory=dict)
    loop_balance_failed: dict = field(default_factory=dict)
    zero_sum_tabu: dict = field(default_factory=dict)

    def clear(self): ...   # прибирає 4 рядки .clear() у BASIN_ESCAPE
```

Зауваження: зараз `self.is_late_game` (порог 0.5, рядок 366) і локальна `is_late_game = self.progress_ratio >= 0.5` (рядок 653) — дві різні змінні з майже однаковим змістом і різним оператором порівняння. `@property` знімає таку можливість.

### 4.4 `run` (191 рядок): витягнути міграцію

`run` читається добре до рядка 398; далі 60 рядків вбудованої логіки міграції (три гілки за температурою: soft injection / hybrid / exploration shield). Це `_migrate(shared_progress, gb)` — окремий метод на 60 рядків, після якого `run` стає ≈ 90 рядків і читається як опис раунду:

```python
while self._within_budget(start_time, time_budget, epoch_start_sims):
    self._sync_with_peers(shared_progress)
    self._check_mini_restart(gb)
    self._apply_swap(round_idx, shared_progress)
    if self.state.stagnation_counter >= 2:
        self._apply_kick(round_idx, shared_progress)
    next_gen = self._generate_mutations()
    just_flushed = self._select_next_beam(next_gen, just_flushed, ...)
    self._maybe_migrate(shared_progress, round_idx)
    round_idx += 1
```

Також `_spatial_crossover` містить `import random` / `import networkx as nx` у тілі функції (рядки 1033-1034) при тих самих імпортах на початку файлу — прибрати.

### 4.5 Один модуль запуску воркера замість трьох

```
analytical_solver/
└── worker.py     run_island_worker(task) + WorkerLogger + worker_init + get_temp_root
```

`main.py` і `gui.py` імпортують `run_island_worker` і передають його як `func` у `pool.apply_async` — мавпячий патч `AnalyticalSolver.worker_task = staticmethod(...)` видаляється, `worker_task` як classmethod видаляється, обидві копії `analytical_worker_task` видаляються. Замість `sys.modules[...]`-магії — звичайні `from .context import SolverContext`.

Заодно замінити позиційний кортеж на 12 елементів:

```python
# було — args[0]...args[11], з двома `if len(args) > N` на хвості
diams = args[0]; v_opt = args[1]; time_budget = args[2]; ...

# стало
@dataclass(frozen=True)
class WorkerTask:
    diameters: list
    v_opt: float
    time_budget: float
    global_best_cost: float
    global_archive: list
    seed: int
    worker_id: int
    log_dir: str | None
    epoch: int
    max_sims: float = float('inf')
    n_workers: int = 1
```

### 4.6 GUI: три файли замість одного на 1 330 рядків

`PipelineOptimizerApp` — 1 106 рядків і 25 методів, що змішують чотири різні відповідальності. Розділення:

```
runner.py               OptimizationRunner — сценарій запуску без жодного tkinter,
                        спільний для main.py і gui.py (прибирає §3.1 повністю)
netdraw.py              draw_network(ax, wn, diameters_by_pipe, *, units, show_legend, title)
                        + diameter_column(df) — прибирає 4 копії з §3.2
gui/app.py              PipelineOptimizerApp   ≈ 350 рядків (UI + обробники подій)
gui/results_viewer.py   ResultsViewer          ≈ 300 рядків (open_results_viewer + load_viewer_data
                                                + _on_log_select — це окреме вікно, не частина головного)
```

`OptimizationRunner` — ключовий крок. Зараз `run_optimization` (GUI) і `main()` (CLI) — це один алгоритм, записаний двічі:

```python
# runner.py
@dataclass
class RunRequest:
    inp_file: str
    cost_file: str
    h_min: float
    units: str
    v_opt: float
    runs: int
    cores: int
    max_sims: int | None
    run_mode: str          # "analytical" | "fast_analytical"

class OptimizationRunner:
    def __init__(self, request: RunRequest, on_progress=None, should_stop=None): ...
    def run(self) -> list[RunResult]: ...        # ~120 рядків, без UI
    def export(self, results, base_dir): ...     # ~40 рядків, без UI
```

GUI тоді: зібрати `RunRequest` із полів, передати `on_progress=self.sync_ui_state`, `should_stop=lambda: self.is_stopped`. CLI: зібрати `RunRequest` з `argparse`. `main.py` скорочується з 317 до ~80 рядків, `run_optimization` — з 171 до ~25.

Два супутні дефекти GUI, які видно при такому розділенні:

* `self.is_stopped`, `self.current_pool`, `self.last_run_dir`, `self.highest_sims`, `self._last_drawn_cost_for_map` не ініціалізуються в `__init__` — вони створюються в `start_thread`/`run_optimization`. Натискання «ЗУПИНИТИ» до першого запуску йде в `stop_thread:435` → `if self.is_stopped` → `AttributeError`. Кнопка вимкнена, тому зараз недосяжно, але стан класу залежить від порядку натискань. Усі поля — в `__init__`.
* `_setup_ui` (131 рядок) розбити на `_build_sidebar`, `_build_main_view`, `_build_log_panel` — механічно, за наявними коментарями-розділювачами.

---

## 5. Цільова структура

```
GAPipeDiameterSearch/
├── main.py                     ~80 рядків: argparse → RunRequest → OptimizationRunner
├── runner.py                   ~180: сценарій запуску, спільний CLI/GUI
├── config.py                   ~40: SolverConfig (замість GAConfig, без GA-полів)
├── cost_table.py               ~45: завантаження цін (замість ga_data.py)
├── utils.py                    ~35: логери, format_time (замість ga_utils.py)
├── water_sim.py                ~180: WaterSimulator + SimStats
├── netdraw.py                  ~90: єдине малювання мережі
├── plot.py                     ~180: convergence + export_solution (карта → netdraw)
├── analytical_solver/
│   ├── __init__.py
│   ├── cache.py                17
│   ├── fast_math.py            85
│   ├── context.py              ~170: + h_min, score(), stats()
│   ├── pool.py                 ~60
│   ├── local_search.py         ~230: gradient_squeeze розбити на _try_downgrade/_try_upgrade/_pass
│   ├── kicks.py                ~520: 10 стратегій → KickResult
│   ├── kick_registry.py        ~60: KickSpec, KICK_REGISTRY, KickResult
│   ├── strategy_selector.py    ~110: UCB1 / VND
│   ├── acceptance.py           ~130: політика прийняття
│   ├── mutation_rates.py       ~60: JADE
│   ├── seeds.py                ~130: SeedFactory (без мертвих методів)
│   ├── island_worker.py        ~350: IslandWorker
│   ├── worker.py               ~90: run_island_worker, worker_init, WorkerTask
│   └── solver.py               ~200: AnalyticalSolver (solve_fast + solve_standalone)
├── gui/
│   ├── app.py                  ~350
│   ├── results_viewer.py       ~300
│   └── streams.py              ~30: GUIStream
├── configs/                    Hanoi_1M.json, balerma_15M.json
├── experiments/archive/        ← нинішня tests/
├── tests/                      ← новий, реальні pytest-тести
├── InputData/
└── attic/                      Old/ + make_warm_seeds, якщо не повертати епохи
```

Жодного файлу понад ~520 рядків, жодного методу понад ~60.

---

## 6. Поетапний план

| Етап | Дія | Δ рядків | Ризик | Час |
|---|---|---|---|---|
| **0** | Закомітити поточний стан. Прогнати Hanoi і Balerma з фіксованим seed, зберегти еталонні `solution_champion.csv` — це регресійний тест на всі наступні етапи | — | — | 1 год |
| **1** | Видалити `analytical_solver.py`; прибрати `GeneticOptimizer`, `worker_eval_task`, `worker_eval_wrapper`, гілку `"ga"`, невикористані імпорти/змінні; `ruff --fix`; винести `build/`, `dist/`, `temp.*`, `text.txt`; `tests/` → `experiments/archive/`; `Old/` → `attic/` | **−1 640** | нульовий | 1 год |
| **2** | Виправити три дефекти: `add_basin_to_tabu` замість `basin_tabu.append` (pool), `SimStats` замість `feas`=`max_p`, ініціалізація полів GUI у `__init__`. Перепрогнати еталон — **тут результати зміняться**, зафіксувати новий еталон | ≈ 0 | середній (змінює поведінку — свідомо) | 2 год |
| **3** | `ctx.h_min` + `ctx.score()`; `KickResult`; `KICK_REGISTRY` | −120 | низький | 3 год |
| **4** | `worker.py` — один запуск воркера; видалити мавпячий патч і дві копії `analytical_worker_task` | −210 | низький | 2 год |
| **5** | `runner.py` — `OptimizationRunner`; `main.py` і `gui.run_optimization` переходять на нього | −250 | середній | 4 год |
| **6** | `netdraw.py` — прибрати 4 копії малювання | −120 | низький | 2 год |
| **7** | Розбити `_apply_kick`: `StrategySelector`, `AcceptancePolicy`, `AdaptiveRates` | −40, CC 181 → 12 | **високий** — серце алгоритму, тільки після етапу 0 | 6 год |
| **8** | `WorkerConfig` / `SearchState` / `KickMemory`; прибрати 16 `getattr`/`hasattr` | −60 | середній | 3 год |
| **9** | Видалити машинерію епох за списком §2.3 (підтверджено як історичний артефакт: епохи замінені однією епохою + острівна модель з асинхронною синхронізацією). `make_warm_seeds` і `greedy_path_relink` → `attic/`, решта — видалити й перейменувати | **−150** | низький (код недосяжний) | 2 год |
| **9а** | *Опційно, як експеримент, не як рефакторинг:* `make_island_seeds` — розподіл свіпу швидкостей між островами замість однакового старту (§2.3.1). Це зміна алгоритму → окремий комміт, окремий прогін, ablation-вісь | +15 | — | 2 год |
| **9б** | *Опційно:* прибрати асиметрію послідовного режиму (§2.3.2) — локальний `dict` замість `manager.dict()` при одному воркері | +10 | низький | 1 год |
| **10** | `gui/` на три файли; `_setup_ui` на три методи | ≈ 0 | низький | 3 год |
| **11** | Перейменувати `ga_config`→`config`, `ga_data`→`cost_table`, `ga_utils`→`utils`, `GAConfig`→`SolverConfig` | ≈ 0 | низький | 1 год |
| **12** | pytest: `StrategySelector`, `AcceptancePolicy`, `SolutionPool`, `LRUCache`, `cost_table`, `ctx.score` — усе без EPANET | +250 | — | 4 год |

Разом: **≈ −2 100 рядків** (6 531 → ≈ 4 400 плюс 250 тестів) і жодного файлу з MI = 0.00.

Порядок важливий: етапи 1–2 дешеві й дають найбільший ефект на «проглядовість» (зникає подвійний пошук по проєкту). Етап 7 — найдорожчий і найризикованіший, тому після нього обов'язкове порівняння з еталоном етапу 2 на обох мережах.

---

## 7. Якщо часу мало: 90 хвилин з максимальним ефектом

1. Спершу `git add -A && git commit` (файл не покритий `.gitignore`, але переконайтесь, що він відстежується), потім `git rm analytical_solver.py` — −1 560 рядків, пошук по проєкту перестає давати по два результати на кожен метод. **Це одна дія, що вирішує половину скарги «код виглядає масивним».**
2. `ruff check --fix .` + видалити `GeneticOptimizer`, `worker_eval_task`, `worker_eval_wrapper`, гілку `"ga"`, `corridor_pool_streak`, `_last_global_improvement_sim`, `baseline_bonus`, `_build_diverse_archive`, `global_failed_basins`.
3. `self.ctx.h_min = simulator.config.h_min` у `SolverContext.__init__` + заміна 43 входжень `self.ctx.simulator.config.h_min` → `self.ctx.h_min`.
4. `orchestrator.py:821` → `self.pool.add_basin_to_tabu(self.run_best_sol)` — вмикає basin tabu на Balerma.
5. `tests/` → `experiments/archive/`, `build/`+`dist/` з робочої теки.

---

## 8. Побічне: код і документація розійшлися

`readme.md` (52 КБ) описує алгоритм добре, але вже не описує цей код:

* §4 документує `baseline_bonus` — поле мертве;
* §6 документує систему каст і `greedy_path_relink` як робочу частину seeding — це артефакт багатоепохної версії, недосяжний при `epochs = 1`. Замінити описом того, що фактично виконує цю роль зараз: асинхронна синхронізація островів (`_process_ipc`, `_check_rescue`, міграція зі `_spatial_crossover`, `_check_mini_restart`). Це найважливіша правка документації: зараз §6 описує архітектуру, від якої ви відмовились, і саме її прочитає рецензент;
* §5 описує `basin_tabu` як робочу FIFO-перевірку — не працює для мереж > 50 труб;
* §2 «Penalized score» описує одну формулу — у коді чотири її варіанти;
* ніде не зафіксовано, що послідовний режим не має архіву, міграції та BASIN_ESCAPE (§2.3.2) — а це впливає на порівнянність baseline.

Після етапу 2 і 3 варто синхронізувати ці чотири місця: у дисертації та статтях ви посилаєтесь на документований алгоритм, і розбіжність між описом і кодом — це саме те, що ставлять у претензію на захисті.

---

## 9. Команди, які треба виконати вручну (етап 1)

Робочий простір Claude не має доступу до файлової системи ПК через оновлення Windows від 8 вересня — правки вмісту файлів записані, але видалення файлів, `git mv` і `git rm` доступні лише вам. У `cmd` з кореня проєкту:

```bat
cd /d E:\Projects\Python\GAPipeDiameterSearch

:: 0. зафіксувати стан ДО (включно з уже записаними правками етапу 1)
git add -A
git commit -m "baseline before refactoring + stage 1 cleanup"

:: 1. головне: недосяжний моноліт
git rm analytical_solver.py

:: 2. звільнити ім'я tests/ і прибрати Old/ з кореня
mkdir experiments
git mv tests experiments\archive
git mv Old attic

:: 3. конфіги запусків
mkdir configs
git mv Hanoi_1M.json configs\Hanoi_1M.json
git mv balerma_15M.json configs\balerma_15M.json

:: 4. сміття (не відстежується git, тому просто видалити)
rmdir /s /q build dist _temp_sim_data
del temp.bin temp.inp temp.rpt text.txt

git add -A
git commit -m "stage 1: drop unreachable analytical_solver.py, rename tests/, archive Old/"
```

Якщо `git rm` або `git mv` скаржиться `did not match any files`, шлях не відстежується git — тоді для нього просто `del` / `move` замість git-команди.

**Перевірка після цього:**

```bat
python main.py --inp InputData\Hanoi\Hanoi.inp --costs InputData\Hanoi\costs.csv --run_mode fast_analytical --units in
```

Має вивести `FINAL RESULT: ~6.7M$` і `OK`. Якщо після видалення `analytical_solver.py` щось падає з `ImportError` — значить десь лишилося посилання на моноліт; надішліть traceback.

Після видалення `tests/` перевірте також `.vscode/settings.json` (105 байт) — якщо там прописано `python.testing` з шляхом `tests`, його треба поправити.
