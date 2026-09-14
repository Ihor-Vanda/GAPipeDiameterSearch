# Ревізія коду ПАК підтримки прийняття рішень (GAPipeDiameterSearch)

Дата аналізу: 13.09.2026. Проаналізовано 16 Python-файлів робочого коду (6 531 рядок), без папки `Old/`.
Інструменти: AST-аналіз, `ruff` (pyflakes/ARG/ERA), `radon` (цикломатична складність, MI), порівняння дублікатів через `difflib`.

---

## 0. Стан виконання

| Етап | Стан | Примітка |
|---|---|---|
| 1 — видалення мертвого коду | **зроблено 13.09** | 10 файлів правлено; файли/папки видалені вручну (§9) |
| 0 — відтворюваність + страхова сітка | **зроблено 13.09** | додано `--seed`; 26 тестів, 17 с, без EPANET (§10) |
| — виправлено попутно | **13.09** | послідовна гілка `solve_standalone` не повертала рішення (§2.3.2) |
| 2 — три дефекти | **зроблено 13.09** | еталон Hanoi збігся до цифри; 37 тестів (див. нижче) |
| 3 — KickResult + реєстр + score | **зроблено 13.09** | §13; 5/8 seed змінились (свідомо, рішення §12) |
| SUBMARINE | **закрито 13.09** | §15–16; 124 занурення, improved=0; рекомендація — revert із тегом |
| 4 — один модуль запуску воркера | **зроблено 13.09** | §17; еталон збігся точно, 84 тести |
| 5 — спільний сценарій запуску | **зроблено 13.09** | §18; 126 тестів; еталон «до/після» треба звірити на вашій машині (§18.4) |
| 6 — `netdraw.py` | **зроблено 14.09** | §19; 4 копії → 1; PNG-звіт розійшовся на 11 пікселів із 17.6 млн |
| 7 — розбір `_apply_kick` | **2 з 3** | §20; CC 149 → 106; лишилась `AcceptancePolicy` |
| 8–12 | попереду | тести на кожен видобутий модуль — див. §10, шар 3 |

**Еталон (уточнено 14.09).** Числа 6.1182 / 6.1090, якими я користувався до етапу 5, були **артефактом часового обрізання**: прогін зупинявся за годинником на ~60 тисячах симуляцій, і 6.1182 — це просто точка на траєкторії пошуку. Після A/B на трьох комітах справжній еталон такий:

| Команда | Результат | Симуляції |
|---|---|---|
| `--units in --cores 1 --seed 42` | **6.0812 M$**, P=30.01 м | **1 001 296** |

Це число отримане **побітово однаковим** на етапах 4 (`20fd4eb`), 5 (`0e9313b`) і 6 (`ef928c6`), на машині користувача і в контейнері — включно з однаковою кількістю симуляцій. Тобто прогін повністю визначається `--seed`, а не швидкістю машини: часовий ліміт не встигає зіграти, бо бюджет симуляцій вичерпується раніше. **Будь-яке розходження з цими двома числами на наступних етапах = регресія.**

Зверніть увагу: `--units in` обов'язкове для Hanoi (таблиця цін у дюймах); чому помилка в цьому прапорці коштує 10 хвилин і не діагностується — §18.5.

**Етап 2, фактично зроблено.** Усі три дефекти виправлені; еталонні прогони Hanoi того часу збіглися точно (6.1182 / 6.0812 / 6.8013 M$ для `--cores 1 --seed 42`, `--seed 7` і `fast_analytical`).

Це очікувано і не означає, що правки нічого не дали: Hanoi має 34 труби, а `get_basin_signature` для мереж ≤ 50 труб повертає сам вектор рішення — там підпис і вектор збігаються, тому дефект basin tabu на Hanoi був невидимий. Ефект з'явиться на Balerma (454 труби), де підпис стискається у ~25 чанків. **Тому перший прогін Balerma після етапу 2 законно дасть інші числа, ніж раніше — це і є виправлення, а не регресія.**

**Дефект 1 — basin tabu.** `orchestrator.py`: `self.pool.basin_tabu.append(tuple(self.run_best_sol))` → `self.pool.add_basin_to_tabu(self.run_best_sol)`. Тепер у deque кладеться підпис басейну — те саме, що шукає `is_basin_tabu()`.

**Дефект 2 — `SimStats`.** Створено `sim_stats.py` — предметний тип без залежностей (окремо від адаптера до EPANET, тому солвер і тести працюють із ним без імпорту wntr). `WaterSimulator.get_stats()` повертає `SimStats(cost, p_min, p_max, crit_node)`; це `NamedTuple`, отже старе позиційне розпакування працює без змін.

Головне — прибрано саму пастку: змінної `feas`, у яку розпаковувався `p_max`, більше **не існує**. У 22 місцях `c, p, feas, _ = ...` стало `c, p, _, _ = ...`, а перевірки `if feas and p >= h_min` / `if not feas or p < h_min` скоротились до `if p >= h_min` / `if p < h_min`. Це доказово еквівалентно при `h_min > 0`: `p_max >= p_min`, тож якщо `p_min >= h_min > 0`, то `p_max` завжди truthy, а при збої симуляції `p_min = -1.0` і друга умова відсікає результат сама.

Шість місць, де `feas` справді використовувався як прапорець (і тому нічого не перевіряв), переведені на явні `st.simulation_ok` / `st.is_feasible(h_min)` / `st.surplus(h_min)`: `_apply_kick` (deficit), `_generate_mutations`, `_beam_search_and_update`, `_emergency_pool_diversity`, `_emergency_respawn` (×2).

Заодно додано `SolverContext.h_min` і прибрано ланцюг `self.ctx.simulator.config.h_min` через три об'єкти — 43 входження → 0.

**Дефект 3 — поля GUI.** `is_stopped`, `current_pool`, `last_run_dir`, `highest_sims`, `_last_drawn_cost_for_map`, `_last_drawn_sol`, `_live_inp_path`, `topo_cb`, `available_logs` тепер ініціалізуються в `__init__`. Шість `getattr(self, ..., default)` замінено на звичайні звернення, а `command=getattr(self, 'on_mode_change', None)` — на пряме `command=self.on_mode_change`.

**Тести: 26 → 37.** Додано `tests/test_sim_stats.py` (5) і `tests/test_context.py` (5); `tests/test_pool.py` перероблено — характеризаційний тест дефекту замінено на два тести правильної поведінки (`add_basin_to_tabu` пише підпис і дедуплікує).

**Що свідомо НЕ зроблено на цьому етапі** (щоб не змішувати виправлення з рішеннями про алгоритм):

* **Гілка HYBRID SUCCESS** (`orchestrator.py`, міграція при `T < 0.7`) приймає гібрид **лише за вартістю, без перевірки тиску** — так було й до етапу 2, бо `feas` там нічого не перевіряв. Поведінка збережена, у коді стоїть явний комментар. Виправляти це = змінювати правило прийняття, отже треба окремий прогін і рішення. Кінцевий `FINAL GUARD` у `run()` недопустиме рішення все одно лікує, тому це не критично.
* **Уніфікація формули penalized score.** У `local_search` score = `c - surplus * dyn_bonus` без правила «надлишок > 10 м → бонус ×0.2», яке є в `orchestrator`. Звести їх до одного `ctx.score()` (§4.2) — це **зміна поведінки локального пошуку**, тому вона належить етапу 3 як окреме рішення, а не як рефакторинг.

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

Тобто послідовний прогін — це не «той самий алгоритм на одному ядрі», а суттєво інший алгоритм.

**Уточнення після перевірки прогонами (13.09):** послідовна гілка ще й недосяжна з обох точок входу — `main.py:234` і `gui.py:495` створюють `multiprocessing.Pool` завжди, навіть при `--cores 1`, тому `self.mp_pool` ніколи не `None`. А коли її таки викликати напряму (`AnalyticalSolver(..., pool=None)`), виявляється, що вона **взагалі не повертає знайдене рішення**:

```python
# solve_standalone, рядки 1498 і 1512 — обидві гілки наповнюють epoch_results
if sol is not None: epoch_results.append((c, sol))
```

а далі `epoch_results` **ніде не читається**. У паралельному режимі це нешкідливо, бо результати приходять через `shared_progress`; у послідовному `shared_progress is None`, тому рішення воркера втрачається і `solve_standalone` доходить до `[WARNING] No valid solution found. Returning safe default.` і повертає «усі максимальні діаметри».

Перевірено на фейковій мережі: воркер знаходив 0.4600 M$, а `solve_standalone` повертав 1.3800 M$ — рівно baseline «усі максимальні».

Виправлено (5 рядків): перед агрегацією з `shared_progress` додано зведення `epoch_results` у `global_best_cost`/`global_best_sol`. Для паралельного режиму це no-op (ті самі значення, беремо мінімум), для послідовного — робить його працездатним. Це прибирає і мертвий стан `epoch_results` із §2.5.

Залишається зробити `--cores 1` справді послідовним (щоб `pool=None` і був детермінізм) — або окремим прапорцем, або локальним `dict` замість `manager.dict()` при одному воркері.

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
| `epoch_results` у `solve_standalone` | 2 (1498, 1512) | 0 → **виправлено 13.09**, див. §2.3.2: у послідовному режимі це ламало результат |

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

Робочий простір Claude не має доступу до файлової системи ПК через оновлення Windows від 8 вересня — правки вмісту файлів записані, але видалення файлів, `git mv` і `git rm` доступні лише вам.

Команди нижче — для **Git Bash (MINGW64)**, це робоча оболонка проєкту. Увага: `rmdir` у Git Bash — POSIX-утиліта, вона видаляє лише порожні каталоги і не розуміє прапорців `cmd` (`/s /q`); у PowerShell `rmdir` — це псевдонім `Remove-Item`, де аргументи розділяються комами. Тому нижче `rm -rf`.

```bash
cd /e/Projects/Python/GAPipeDiameterSearch

# 0. зафіксувати стан ДО (включно з уже записаними правками етапу 1)
#    за потреби спершу окрема гілка: git switch -c refactor/stage-1
git add -A
git commit -m "baseline before refactoring + stage 1 cleanup"

# 1. головне: недосяжний моноліт
git rm analytical_solver.py

# 2. звільнити ім'я tests/ і прибрати Old/ з кореня
mkdir -p experiments
git mv tests experiments/archive
git mv Old attic

# 3. конфіги запусків
mkdir -p configs
git mv Hanoi_1M.json configs/Hanoi_1M.json
git mv balerma_15M.json configs/balerma_15M.json

# 4. сміття (git його не відстежує)
rm -rf build dist _temp_sim_data
rm -f temp.bin temp.inp temp.rpt text.txt

git add -A
git commit -m "stage 1: drop unreachable analytical_solver.py, rename tests/, archive Old/"
```

Якщо `git rm` або `git mv` скаржиться `did not match any files`, шлях не відстежується git — тоді для нього просто `rm` / `mv` замість git-команди.

`rm -rf build dist` видалить і скомпільований `gui.exe`; відновлюється через `pyinstaller gui.spec` (файл спеки лишається).

**Перевірка після цього:**

```bash
python main.py --inp InputData/Hanoi/Hanoi.inp --costs InputData/Hanoi/costs.csv --run_mode fast_analytical --units in
```

Має вивести `FINAL RESULT: ~6.7M$` і `OK`. Якщо після видалення `analytical_solver.py` щось падає з `ImportError` — значить десь лишилося посилання на моноліт; надішліть traceback.

Після видалення `tests/` перевірте також `.vscode/settings.json` (105 байт) — якщо там прописано `python.testing` з шляхом `tests`, його треба поправити.

Побічно: перейменування каталогу `tests/` знімає неоднозначність із гілкою `tests` — зараз `git checkout tests` може означати і гілку, і шлях.

---

## 10. Тести: коли і які

Коротка відповідь: **разом із рефакторингом, по одному набору на етап** — не до і не після.

Чому не «зараз для всього коду». Юніт-тест фіксує інтерфейс. Більша частина інтерфейсів, які варто було б тестувати, на етапах 3–8 зникає: `_apply_kick` на 350 рядків неможливо осмислено покрити тестом, контракт кіків зміниться на `KickResult`, `IslandWorker.__init__` втратить 4 параметри. Тести, написані до цього, довелося б переписувати разом із кодом — і вони не ловили б регресії, а лише дублювали б роботу.

Чому не «після всього». Найризикованіший етап — 7 (розбиття `_apply_kick`, CC 181). Робити його без страхової сітки означає покладатись на порівняння з еталонним прогоном вручну.

Правильний розподіл — три шари:

**Шар 1: наскрізна регресія — потрібна ДО етапу 2, вже зроблена.**

`tests/test_solver_regression.py` проганяє `AnalyticalSolver` на `FakeSimulator` і перевіряє: результат допустимий за тиском, дешевший за baseline «усі максимальні», і — головне — **однаковий seed дає однакове рішення**. Останнє переживає перебазування: після етапу 2 числа зміняться, а властивість детермінізму лишиться, тому тест не доведеться правити.

**Шар 2: юніт-тести того, що вже стабільне — зроблено.**

`LRUCache`, `SolutionPool`, `LocalSearch` рефакторинг не змінює по суті, тому їх можна покривати одразу:

| Файл | Тестів | Що покриває |
|---|---|---|
| `tests/conftest.py` | — | `FakeSimulator` + `FakeConfig`, фікстури `sim` і `ctx` |
| `tests/test_cache.py` | 5 | витіснення LRU, оновлення давності при читанні |
| `tests/test_pool.py` | 9 | точне tabu та tenure, відстань Хеммінга, підписи басейнів, `clear_all` |
| `tests/test_local_search.py` | 8 | `gradient_squeeze` (зниження вартості, блокування, незмінність входу, `classical_descent`), `heal_network`, `evaluate_candidate`, `get_high_impact_pipes` |
| `tests/test_solver_regression.py` | 4 | наскрізні прогони, детермінізм за seed |

Разом 26 тестів, **17 секунд, без EPANET і без .inp-файлів**.

Ключ до цього — `FakeSimulator` у `conftest.py`: він реалізує той самий інтерфейс, що `WaterSimulator` (`get_stats`, `get_heuristics`, `get_hydraulic_state`, `graph`, `lengths`, `costs`, `sources`, `config`), але на аналітичній формулі втрат для послідовної мережі R → J1 → … → Jn. Коефіцієнт підібраний так, щоб обмеження реально працювало: «всі мінімальні» недопустиме (−93.97 м), «всі максимальні» допустиме (93.37 м). Без цього класу тестувати солвер довелося б через EPANET — повільно, з тимчасовими файлами і без можливості поставити в CI.

Запуск:

```bash
pytest                    # усе, ~17 с
pytest -m "not slow"      # лише юніт-тести, ~1 с
```

**Шар 3: юніт-тести на кожен видобутий модуль — під час етапів 3–8.**

Правило: етап не закритий, поки видобутий клас не має тестів. Це не додаткова робота, а те, що робить видобування осмисленим — зараз ці алгоритми неможливо перевірити інакше, ніж повним прогоном:

| Етап | Що з'являється | Що тестувати без EPANET |
|---|---|---|
| 3 | `KickResult`, `KICK_REGISTRY` | повнота реєстру, доступність стратегій за температурою, `failed` при `solution is None` |
| 3 | `ctx.score()` | правило «надлишок > 10 м → бонус ×0.2», `inf` при дефіциті |
| 7 | `StrategySelector` | UCB1 обирає найкращу за нагородою, `mark_failed` виводить із пулу після `max_fails`, burst-penalty не дає повторювати ту саму, VND-фолбек при `is_late_game` |
| 7 | `AcceptancePolicy` | всі п'ять вердиктів: рекорд, basin escape, water-level, HAIL MARY, відхилення; поріг «вибуху вартості» |
| 7 | `AdaptiveRates` | JADE: зростання при успіху, стиснення ×0.90 при вибуху, межі |
| 8 | `SearchState`, `KickMemory` | `is_late_game` як property, `clear()` |
| 5 | `OptimizationRunner` | зворотне відображення діаметрів у індекси, вибір чемпіона серед `runs` |
| 6 | `netdraw.draw_network` | вибір колонки діаметра, ширини/кольори при однакових діаметрах, порожній список |
| 11 | `cost_table.load_config` | розділювачі `,`/`;`, альтернативні назви колонок, mm vs in |

Етап 12 у §6 після цього зводиться до дописування того, що лишилось, а не до написання всього з нуля.

### 10.1 Передумова, якої не було: відтворюваність

Етап 0 плану («еталонні прогони з фіксованим seed») був **нездійсненний** — у коді не було жодної точки, де сіється RNG головного процесу:

* `solve_fast` виконується цілком у головному процесі й ніколи не сіявся → повністю стохастичний;
* `seed_modifier = random.randint(1, 10000)` (`solve_standalone`) брався з **незасіяного** RNG, тому і воркери щоразу отримували інші seed.

Тобто жоден прогін не був відтворюваний, і два запуски з однаковими параметрами законно давали різні числа (6.7588 vs 6.6561 M$ на Hanoi).

Додано `--seed` у `main.py` — одна точка входу, від якої залежить і `solve_fast`, і `seed_modifier` воркерів. Перевірено:

| Режим | Прогін 1 | Прогін 2 | Висновок |
|---|---|---|---|
| `fast_analytical --seed 42` | 6.8013 M$, 1852 сим. | 6.8013 M$, 1852 сим. | відтворюваний |
| `fast_analytical --seed 7` | 6.6821 M$ | — | seed справді впливає |
| `fast_analytical` без seed | 6.8922 M$ | 6.8433 M$ | як і було |
| `analytical --cores 1 --seed 42` | 6.1182 M$ | 6.1182 M$ | вартість відтворювана |
| `analytical --cores 3 --seed 42` | 6.2152 M$ | 6.1996 M$ | **не** відтворювана |

Останній рядок — не дефект, а властивість асинхронної острівної моделі: `_process_ipc`, `_check_rescue` і міграція читають `shared_progress` у момент, який залежить від реального планування процесів. Зафіксувати це можна лише прив'язавши синхронізацію до лічильника симуляцій замість часу — окрема задача, і не факт що потрібна.

Практичний висновок для дисертації та для рефакторингу:

* **точний еталон** (побітове порівняння до/після етапу) — `--cores 1 --seed S` плюс `pytest tests/test_solver_regression.py`;
* **статистичний baseline** (для таблиць у статтях) — N прогонів паралельного режиму з різними seed, mean / median / best / std. Зараз `--runs N` дає саме це, але всі прогони беруть seed з одного незасіяного джерела; варто зробити `seed_i = args.seed + i`, щоб серія була відтворювана цілком.

При `--cores 1` вартість збігається побітово, але лічильник симуляцій відрізняється на ~0.5 % (41 084 vs 40 893): частина його збирається через `update_progress_ping`, що спрацьовує за таймером. На результат це не впливає, але як метрику в таблицях краще брати `ctx.sim_count`, а не це зведене число.

---

## 11. Команди для етапу 2 (Git Bash)

Файли вже записані на диск. Лишилось закомітити:

```bash
cd /e/Projects/Python/GAPipeDiameterSearch

# перевірка перед коммітом
pytest -q                         # має бути 37 passed
python main.py --inp InputData/Hanoi/Hanoi.inp --costs InputData/Hanoi/costs.csv \
               --run_mode analytical --units in --cores 1 --max_sims 40000 --seed 42
# очікується FINAL RESULT: 6.1182M$ — точний збіг із еталоном

git add -A
git commit -m "stage 2: fix basin tabu signature, introduce SimStats, init GUI state in __init__"
```

Якщо `pytest` не встановлений: `pip install pytest`.

**Прогін Balerma** варто зробити саме зараз, бо це перший випадок, де виправлення basin tabu дає інший результат:

```bash
python main.py --inp InputData/Balerma/Balerma.inp --costs InputData/Balerma/costs.csv \
               --run_mode analytical --units mm --cores 5 --max_sims 15000000 --seed 42
```

(перевірте `--units` і `--hmin` за вашими налаштуваннями для Balerma — у `InputData/Balerma/min_p.txt` лежить потрібний тиск). Цей результат стає новим еталоном Balerma для етапів 3–12.

---

## 12. Формула score: замір, а не думка

Питання було — уніфікувати `local_search` з правилом «надлишок > 10 м → бонус ×0.2», чи лишити два варіанти. Замість аргументів зроблено замір.

### 12.1 Скільком обчисленням взагалі стосується демпфування

Інструментований прогін Hanoi (`--cores 1`, 40 k симуляцій, seed 42), розподіл запасу тиску на 8 893 виклики score:

| Запас над h_min | Викликів | Частка |
|---|---|---|
| < 0 (недопустиме) | 815 | 9.16 % |
| 0–1 м | 7 331 | **82.44 %** |
| 1–5 м | 681 | 7.66 % |
| 5–10 м | 43 | 0.48 % |
| 10–30 м ← демпфування | 23 | **0.26 %** |
| > 30 м ← демпфування | 0 | 0.00 % |

Отже премія за запас у робочому режимі — справді **тай-брейкер**: 82 % порівнянь мають запас менше метра. Гілка демпфування спрацьовує на 0.26 % обчислень.

### 12.2 Порівняння трьох політик

Hanoi, `--sequential`, 40 k симуляцій, 8 seed, повністю детерміновано (M$):

| seed | damped | undamped | none |
|---|---|---|---|
| 1 | 6.1883 | 6.1883 | **6.0974** |
| 2 | **6.0974** | **6.0974** | **6.0974** |
| 3 | **6.0812** | **6.0812** | 6.0974 |
| 5 | 6.0961 | 6.0961 | **6.0812** |
| 7 | 6.1054 | 6.1054 | **6.0974** |
| 11 | **6.0974** | **6.0974** | **6.0974** |
| 42 | 6.1923 | 6.1923 | **6.0974** |
| 99 | **6.0961** | **6.0961** | 6.1424 |
| **mean** | 6.1193 | 6.1193 | **6.1010** |
| **stdev** | 0.0444 | 0.0444 | **0.0177** |
| **worst** | 6.1923 | 6.1923 | **6.1424** |

**`damped` і `undamped` дали однаковий результат у всіх 8 seed** — різниця нульова. Це прямо узгоджується з §12.1: гілка демпфування спрацьовує так рідко, що ні разу не змінила траєкторію пошуку.

**Висновок щодо початкового питання: уніфікувати, і це безризиково.** Вибір політики на Hanoi не впливає на результат узагалі, тож єдина причина обирати — послідовність коду й теоретична обґрунтованість:

* в задачі мінімізації вартості обмеження тиску в оптимумі активне, тому необмежена лінійна премія за надлишковий напір — це перекос убік дорожчих рішень; демпфування його обмежує;
* `readme.md` §2 документує **одну** формулу penalized score, а в коді їх було 13 входжень у трьох варіантах (damped у 3 місцях, undamped у 5, окремий epsilon-варіант у beam). Для розділу дисертації з ablation це найгірший стан: описана формула не та, що працює.

Реалізовано як `SolverContext.score(cost, p_min, dyn_bonus)` з параметром `surplus_bonus_policy` ∈ {`damped`, `undamped`, `none`}, за замовчуванням `damped`. Усі 7 обчислень у `local_search` зведені до цього методу (перевірено: прогін Hanoi seed 42 до і після рефакторингу дав ідентичний результат). 8 входжень в `orchestrator` зводяться на етапі 3.

### 12.3 Побічний результат, вартий окремої ablation

`none` — повна відмова від премії — **не була гіршою**: середнє на 0.30 % краще, розкид у 2.5 раза менший, найгірший випадок кращий. Статистично при n = 8 це не значимо (парний t = +1.04, краще у 4 seed, гірше у 2), але напрям цікавий: премія за запас, схоже, дає менше, ніж від неї очікували, і додає розкиду.

Це готова ablation-вісь для дисертації: «чи виправдовує себе тай-брейкер за запасом тиску». Потрібен прогін на Balerma з реальним бюджетом і 20+ seed. Політика вже є параметром, тому нічого доробляти не треба.

### 12.4 Головне, що знайшлося попутно: прогони не були відтворюваними

Спроба зміряти політики виявила, що два запуски з однаковим `--seed` дають різні числа. Причини — три, і всі знайдені та закриті:

**1. Незасіяний RNG головного процесу.** Закрито `--seed` (§10.1).

**2. Часовий бюджет, виведений з калібрування.** `time_limit_sec = (max_sims / cluster_speed) * 5`, де `cluster_speed` міряється годинником при старті. Точка зупинки залежала від завантаження машини. Додано `--time_limit` (0 = без обмеження, зупинка лише за `--max_sims`); `--sequential` вмикає це автоматично.

**3. `PYTHONHASHSEED`.** Порядок обходу множин з рядковими ключами (імена вузлів мережі) Python рандомізує на кожен процес. Замір на Hanoi: 4 прогони з тим самим seed дали 3 різні результати; з `PYTHONHASHSEED=0` — 4 однакові (6.1883 M$, 40 905 симуляцій, до симуляції). Змінну неможливо застосувати після старту інтерпретатора, тому `main.py` при заданому `--seed` перезапускає процес із `PYTHONHASHSEED=0`.

**4. Спільний архів через монітор із таймером** — не закрито і закривати не варто. Монітор у `solve_standalone` оновлює `shared_progress['global_archive']` за таймером, а гілки, що залежать від розміру архіву (`BASIN_ESCAPE` при `>= 2`, consensus locking при `>= 3`), читають його в довільний момент. Тому **паралельний режим невідтворюваний у принципі**, і це нормальна властивість асинхронної острівної моделі.

Додано `--sequential`: один острів у тому ж процесі, без `Pool` і `Manager`, зупинка лише за бюджетом симуляцій. Перевірено — 4 прогони з `--seed 1` дають побітово однаковий результат разом із лічильником симуляцій.

**Практичний висновок для експериментальної частини дисертації:**

| Задача | Режим |
|---|---|
| еталон для порівняння «до/після» етапу рефакторингу | `--sequential --seed S` (побітово) |
| ablation алгоритмічних варіантів | `--sequential`, 20+ seed, парне порівняння |
| таблиці результатів для статей | паралельний режим, N прогонів, mean / median / best / std |

Числа, отримані раніше без `--seed`, не відтворюються — якщо вони вже в тексті статей, варто перегенерувати серію з фіксованими seed.

### 12.5 Новий еталон Hanoi (`--sequential`, `damped`, 40 k симуляцій)

| seed | 1 | 2 | 3 | 5 | 7 | 11 | 42 | 99 |
|---|---|---|---|---|---|---|---|---|
| M$ | 6.1883 | 6.0974 | 6.0812 | 6.0961 | 6.1054 | 6.0974 | 6.1923 | 6.0961 |

Тести: 37 → **44** (додано `tests/test_score.py` на три політики та межу демпфування).

---

## 13. Етап 3: контракт кіків, реєстр стратегій, одна формула score

### 13.1 `KickResult` замість кортежів різної арності

Було: десять методів `KickStrategies` повертали кортежі довжиною 3 або 4, а четвертий елемент означав різне залежно від стратегії. `_apply_kick` розбирав це так:

```python
if not isinstance(res, tuple) or len(res) < 3: return
forced_sol, locked, log_msg = res[0], res[1], res[2]
if len(res) >= 4:
    if strategy in ["BOTTLENECK", "LOOP_BALANCE"]: failed_pipe_id = res[3]
    elif strategy == "TOPO_INV": path_sig = res[3]
    elif strategy in ["RUIN_RECREATE", ...]: used_pct = res[3]
```

Стало: `analytical_solver/kick_registry.py` з `KickResult` — 28 `return` у `kicks.py` переведені на нього, розбір кортежів зник. Поля `failed_pipe` / `path_signature` / `used_pct` тепер названі, а `failed` — властивість, що повторює стару перевірку дослівно (`solution is None or locked is None`), тому семантика не змінилась.

Заодно зник сентинел `-1` для «немає труби»: труба з індексом 0 при перевірці на falsy читалася б як відсутність труби. Тепер `None`.

### 13.2 `KICK_REGISTRY` замість ланцюга з 11 `elif`

Перелік стратегій існував у **трьох** місцях: `self.strategies` у конструкторі, `all_tracked` там же, і три захардкоджені пули в `_apply_kick`. Додавання стратегії вимагало правок у пʼяти місцях.

Тепер джерело одне — `KICK_REGISTRY` з `KickSpec` (імʼя, метод, чи потрібні цикли, який коефіцієнт μ навчає, що кладе в необовʼязкові поля). Диспетчер став одним рядком:

```python
spec = KICK_REGISTRY.get(strategy) or KICK_REGISTRY[FALLBACK_KICK]
res = getattr(self.kicker, spec.method)(kick_target, **kick_args)
```

Температурні пули виділені в чисту функцію `pool_for(temperature, archive_size, is_tree)`. Важливо, що це **три якісно різні набори**, а не діапазони `[min, max]` на кожну стратегію: у гарячому режимі (`T >= 0.90`) набір змінюється цілком і залежить від розміру архіву. Тому пули описані переліком — інакше рефакторинг змінив би алгоритм.

**Доказ нейтральності** — `tests/test_kick_registry.py` фіксує пули дослівно такими, якими вони були захардкоджені, для восьми точок температури й розміру архіву. Порядок елементів теж перевіряється: вибір стратегії йде через `random.choice(...)` і `max(..., key=...)` по цих списках, тому інший порядок означав би іншу траєкторію пошуку.

**Знайдено недиспетчеризовану стратегію.** У `all_tracked` була `"SUBMARINE"`, якої немає ні в жодному пулі, ні в диспетчері — тобто їй марно виділялися ваги й спроби UCB1. Прибрана; тест `test_tracked_has_no_undispatched_strategies` не дасть повторити. **Це не була випадкова назва** — задум і причини, чому реалізація не пішла, розібрані в §14.

Прибрано також 10 захисних `getattr(self, ...)` / `hasattr(self, ...)` у `_apply_kick` — поля (`_last_strategy`, `_strategy_burst_count`, `bottleneck_boosted_pipes`, `_force_flush_next`, `ipc_immunity`, чотири `mu_*`) тепер ініціалізуються в конструкторі. Чотири з тих `hasattr` взагалі нічого не робили: поля вже були в `__init__`. `nx.is_tree()` тепер обчислюється один раз у конструкторі, а не на кожному кіку.

### 13.3 Одна формула penalized score

Виконано рішення §12. Усі **16** обчислень зведені до `ctx.score()`: 7 у `local_search`, 8 в `orchestrator`, 1 у `kicks` (TRIM). Поза цим методом формули більше не існує.

Це **змінює поведінку**, і саме так і планувалось: з восьми входжень в `orchestrator` пʼять були undamped (`_initialize_seeds`, `_process_ipc`, `_check_rescue`, `_check_mini_restart`, `_emergency_pool_diversity`), три — damped. Тепер усі damped.

Єдине місце, яке НЕ пішло через `ctx.score` — epsilon-вікно в `_beam_search_and_update`: там рішення з дефіцитом до −0.5 м приймається зі штрафом `cost + |surplus| * bonus * 50`, а `ctx.score` для відʼємного запасу повертає `inf`. Ця гілка лишилась окремою і явною.

### 13.4 Результат на еталоні

Hanoi, `--sequential`, 40 k симуляцій (M$):

| seed | до етапу 3 | після | |
|---|---|---|---|
| 1 | 6.1883 | **6.1090** | краще |
| 2 | 6.0974 | 6.0974 | збіг |
| 3 | 6.0812 | 6.0812 | збіг |
| 5 | 6.0961 | 6.0961 | збіг |
| 7 | 6.1054 | **6.0974** | краще |
| 11 | 6.0974 | 6.1182 | гірше |
| 42 | 6.1923 | **6.1182** | краще |
| 99 | 6.0961 | 6.1011 | гірше |

Середня зміна **+0.0170 M$ (+0.28 % на користь нового стану)**; змінилось 5 із 8 seed, краще у 3, гірше у 2.

Цікаво, що в §12 уніфікація політики **лише в `local_search`** не змінила жодного seed, а разом з `orchestrator` змінила пʼять. Причина зрозуміла: `local_search` порівнює рішення-сусіди, що відрізняються одним кроком діаметра, тому запас тиску в них майже однаковий; а в `orchestrator` score рахується при вставці в пул — і туди потрапляють seed зі свіпу швидкостей та інжекції global best, які несуть великий запас. Саме там демпфування й працює.

**Новий еталон Hanoi** (`--sequential`, `damped`, 40 k): seed 1 → 6.1090, 2 → 6.0974, 3 → 6.0812, 5 → 6.0961, 7 → 6.0974, 11 → 6.1182, 42 → 6.1182, 99 → 6.1011. Детермінізм перевірено (seed 42 двічі: 6.1182 M$, 40 889 симуляцій).

### 13.5 Метрики

| | до етапу 3 | після |
|---|---|---|
| `_apply_kick` CC | 181 | **149** |
| формул penalized score у коді | 16 у 3 варіантах | **1** |
| переліків стратегій | 3 | **1** |
| `getattr`/`hasattr` на власному стані в `orchestrator` | 10 | **0** |
| тестів | 44 | **64** |

`_apply_kick` усе ще F: 149 — це майже виключно вибір стратегії (UCB1 + VND-фолбек + burst-penalty) і каскад прийняття рішення. Їх розбирає етап 7 (`StrategySelector`, `AcceptancePolicy`, `AdaptiveRates`); реєстр і `KickResult` — передумова для цього, бо саме вони прибрали з методу все, що не є логікою.

### 13.6 Команди

```bash
python -m pytest -q                       # 64 passed
python main.py --inp InputData/Hanoi/Hanoi.inp --costs InputData/Hanoi/costs.csv \
               --run_mode analytical --units in --sequential --max_sims 40000 --seed 42
# очікується 6.1182M$ (НЕ 6.1923 — score уніфіковано, див. §13.4)

git add -A
git commit -m "stage 3: KickResult contract, KICK_REGISTRY, single penalized-score formula"
```

---

## 14. SUBMARINE: занурення за обмеження тиску

**Уточнено автором:** `SUBMARINE` не була випадковим рядком у словнику. Задум — тимчасово знизити вимогу до тиску, дати алгоритму пройти «глибше» через недопустиму область, а потім повернути вимогу, щоб рішення «виплило» в новій точці простору. З реалізацією тоді виникли проблеми.

Це відома техніка з назвою: **strategic oscillation** навколо межі допустимості (Glover), у літературі з обмеженою оптимізацією — **ε-constraint relaxation** зі спадним допуском. Сенс саме той: басейни допустимих рішень у задачі вибору діаметрів розділені недопустимими перешийками, і пройти між ними, не порушуючи обмеження ні на крок, часто неможливо.

### 14.1 Ідея вже живе в коді — у чотирьох місцях і з чотирма різними числами

| Місце | Що робить | Величина допуску |
|---|---|---|
| `orchestrator.py:780` | `relaxed_h_min = base_h_min - 0.5 * T` перед лікуванням кікнутого рішення | до 0.5 м, пропорційно температурі застою |
| `orchestrator.py:830-831` | приймає рішення з дефіцитом, зі штрафом `c + deficit * (run_best * 0.10)` | `deficit <= 0.5 * T` |
| `orchestrator.py:1073-1079` | epsilon-вікно в beam: у пул, але без права на рекорд; штраф `cost + \|surplus\| * dyn_bonus * 50` | фіксовані −0.5 м |
| `water_sim.py:128` | `evaluate(..., epsilon=0.0)` — параметр існує, не використовується | — |

Плюс п'яте: `GAConfig.epsilon_start = 10.0`, `epsilon_end = 0.0` — саме той спадний розклад допуску, з якого все починалося. Поля ніде не читалися, тому прибрані на етапі 1; відновити можна з git.

`readme.md` §2 документує лише одне з цих п'яти («Relaxed-feasibility (epsilon window у beam search)») — інші чотири в документації відсутні.

Отже механізм не відсутній. Він **розсіяний**, з чотирма незалежними допусками й двома різними шкалами штрафу (×0.10 від найкращої вартості проти ×50 від `dyn_bonus`), і без жодного власника.

### 14.2 Чому реалізація не пішла — три конкретні причини, видимі в коді

**1. Немає фази випливання.** Ніщо не змушує повернутися під строге обмеження. Єдине, що відновлює допустимість, — `FINAL GUARD` у самому кінці `run()` (рядок 488), який жорстко лікує фінальне рішення. Тобто занурення, яке не виплило, під час пошуку не виявляється — воно латається наприкінці, а проміжні недопустимі рішення тим часом лишаються в пулі й впливають на все далі.

**2. Немає точки відкату.** Невдале занурення неможливо скасувати: пул уже заповнений рішеннями з дефіцитом, і в них немає терміну придатності. Strategic oscillation вимагає тримати «найкраще допустиме» окремо від «поточної робочої точки». Перше в коді є (шлях до рекорду вимагає `deficit == 0`, рядок 851), другого — немає.

**3. Щит працює проти підводного човна.** `is_ghost_solution` (`context.py:87`) повертає True завжди, коли `p_min < h_min - 0.01`. Під час занурення **кожне** рішення є «ghost» за побудовою. А викликається щит саме на перспективних — там, де виграш перевищує 2 % від найкращого. Тобто хороше рішення, знайдене після занурення, відкидається як ілюзія. Два механізми прямо суперечать один одному, і цього достатньо, щоб реалізація виглядала зламаною.

### 14.3 Тепер це набагато дешевше зробити

Етап 3 дав те, чого не було: **одну точку, де вирішується допустимість** — `ctx.score()`. Тому послаблення тепер додається в одному місці, а не в чотирьох:

```python
# context.py
def score(self, cost, p_min, dyn_bonus, epsilon=0.0):
    surplus = p_min - self.h_min
    if surplus < -epsilon:          # єдине місце, де діє допуск
        return float('inf')
    if surplus < 0:                 # занурення: штраф, але не заборона
        return cost + abs(surplus) * dyn_bonus * self.DIVE_PENALTY
    ...
```

Необхідні умови, кожна з яких відповідає одній із причин §14.2:

```python
@dataclass
class DiveController:
    """Одне джерело допуску ε і явні фази занурення."""
    h_min: float
    max_depth: float = 0.5        # метрів нижче h_min
    max_dive_rounds: int = 8

    surface_point: list | None = None   # куди відкатитись, якщо не виплили
    rounds_under: int = 0

    def epsilon(self) -> float: ...           # 0, якщо не занурюємось
    def begin_dive(self, from_solution): ...  # запам'ятати точку відкату
    def should_surface(self) -> bool: ...     # вичерпано max_dive_rounds
    def surface(self, solution, ls) -> tuple[list, bool]:
        """heal_network + gradient_squeeze під СТРОГИМ обмеженням.
        Повертає (рішення, чи виплило). Якщо ні — відкат до surface_point."""
```

І обов'язково: `is_ghost_solution` мусить знати про ε, інакше щит і далі гаситиме кожне занурення.

**Важлива пастка, яку варто обійти одразу:** ε не можна закладати в кеш. `ctx.get_cached_stats` кешує `SimStats` за вектором рішення; ε впливає лише на **прийняття**, не на гідравліку. Якщо допуск потрапить у кешовані дані, кеш буде отруєний — рішення, оцінене під зануренням, потім читатиметься як допустиме. Це ймовірна причина частини тодішніх проблем.

### 14.4 Де це в плані

Це **не рефакторинг, а алгоритмічне доповнення**, тому не входить в етапи 4–12. Робити його варто окремою гілкою після етапу 7 (коли `AcceptancePolicy` уже буде виділена — саме їй належить ε) або взагалі після етапу 12.

Натомість тепер є чим це виміряти, чого раніше не було: `--sequential --seed` дає побітову відтворюваність, а `ctx.score` — одну точку для перемикача. Готова ablation-вісь на чотири положення:

| Варіант | ε |
|---|---|
| `off` | 0 — строге обмеження завжди |
| `fixed` | стала −0.5 м (поточне epsilon-вікно beam) |
| `temperature` | −0.5 · T (поточне послаблення в `_apply_kick`) |
| `dive` | занурення з явною фазою випливання й відкатом |

Перші три вже є в коді — їх треба лише звести під один параметр. Четвертий і є SUBMARINE. Порівняння на Hanoi і Balerma з 20+ seed — це самостійний результат для дисертації, бо він відповідає на питання, чи взагалі варто проходити через недопустиму область у цій задачі.

`SUBMARINE` свідомо **не** повернута в `KICK_REGISTRY`: назва в словнику статистики без реалізації нічого не робила, лише витрачала ваги UCB1. Повернути її треба разом із `DiveController`, і тоді вона стане не кіком, а режимом прийняття — бо занурення не збурює рішення, воно змінює правило допустимості.

---

## 15. SUBMARINE зібрано і зміряно

Механізм реалізований (`analytical_solver/dive.py`, прапорці `--dive off|relax|dive`, `--dive_depth`, `--dive_rounds`) і перевірений на Hanoi. Режим `off` відтворює еталон §13.4 **побітово**, тому механізм не впливає на нічого, поки його не увімкнули.

### 15.1 Що саме зроблено

Три режими, які й утворюють ablation:

* **`off`** — поступки немає. `extra_tolerance()` завжди 0.
* **`relax`** — стала поступка `depth` метрів, **без** фази випливання й відкату. Навмисно «неповний» варіант: він ізолює вплив самого послаблення від впливу дисципліни.
* **`dive`** — повний цикл: занурення за застоєм (`stagnation >= stag_limit * 2`), обмежена тривалість (`--dive_rounds`), примусове випливання, відкат у разі невдачі, кулдаун між зануреннями.

Додано те, чого не було в попередній реалізації (§14.2):

1. **Фаза випливання** `IslandWorker._surface()` — повертає строге обмеження, бере найдешевше рішення з пулу (воно може бути недопустимим — для цього й занурювалися), лікує і глибоко полірує **без** поступки.
2. **Точка відкату** — якщо не виплили дешевше, пул повертається до збереженої точки, щоб недопустимі рішення не тягнулися далі.
3. **Щит знає про поступку** — `is_ghost_solution(indices, tolerance)`. Без цього щит гасив кожне занурення: рішення з дефіцитом за побудовою виглядало «ілюзією».
4. **Чесний облік заслуги** — див. §15.2.

### 15.2 Перша спроба обліку була неправильною, і це важливо

Початкова телеметрія порівнювала результат випливання з вартістю на момент **початку** занурення і показала `improved=2, gain=0.031M$`. Це завищення: під час занурення звичайний пошук продовжує працювати й може сам покращити рекорд строго допустимим шляхом. Після виправлення — порівняння з рекордом, актуальним **на момент випливання** — `improved` став 0.

Тобто ті «покращення» належали звичайному пошуку, а не зануренню. Без цієї правки висновок був би протилежним і хибним.

### 15.3 Друга спроба була інертною, і це теж важливо

Перший замір глибини дав **ідентичні** числа для 2 м і 10 м — ознака того, що механізм не працює. Причина: поступка була проведена лише в **гейти прийняття**, а `heal_network` усе одно лікував до строгого `h_min`, а `ctx.score` повертав `inf` на будь-який дефіцит. Тобто занурення не занурювалося.

Виправлення — поступка мусить дійти до того, що **створює** рішення, а не лише до того, що їх приймає:

* `ctx.score(cost, p_min, dyn_bonus, tolerance)` — дефіцит у межах поступки не заборонений, а дорогий (`DEFICIT_PENALTY = 50`, успадковано з epsilon-вікна beam);
* `heal_network(..., tolerance)` — лікує до `h_min - ε`, а не до `h_min`;
* `gradient_squeeze(..., tolerance)` — спуск не витягує рішення назад одразу.

Після цього глибина почала впливати: 0.5 / 2 / 10 м дають 10 / 19 / 22 занурень.

**Це та сама помилка, через яку не пішла попередня реалізація**, і я повторив її з першого разу: послаблення, проведене лише в перевірки прийняття, нічого не робить, бо конструктори рішень (`heal_network`, `gradient_squeeze`) продовжують працювати за строгим обмеженням.

### 15.4 Результат: механізм не виправдав себе на Hanoi

Hanoi, `--sequential`, 40 k симуляцій, глибина 2 м, 15 раундів, 8 seed (M$):

| seed | off | dive | relax |
|---|---|---|---|
| 1 | **6.1090** | 6.1865 | 6.2614 |
| 2 | 6.0974 | **6.0966** | 6.2672 |
| 3 | **6.0812** | 6.1388 | 6.3967 |
| 5 | **6.0961** | 6.0974 | 6.2520 |
| 7 | **6.0974** | 6.1195 | 6.2786 |
| 11 | **6.1182** | 6.1301 | 6.2106 |
| 42 | **6.1182** | **6.1182** | 6.2616 |
| 99 | 6.1011 | **6.0974** | 6.3368 |
| **mean** | **6.1023** | 6.1231 | 6.2831 |
| **stdev** | 0.0124 | 0.0302 | 0.0576 |

| | проти `off` | t | краще / гірше |
|---|---|---|---|
| `dive` | −0.34 % | −1.92 | 2 / 5 |
| `relax` | **−2.96 %** | **−7.57** | 0 / 8 |

Усі результати в усіх режимах допустимі за тиском (строгий шлях до рекорду + `FINAL GUARD` працюють).

**Телеметрія: 100 занурень, 93 виплили в допустиме рішення, `improved = 0`.** Жодного разу випливання не побило рекорд, актуальний на той момент.

### 15.5 Що з цього випливає

**Головний результат — кількісна ціна дисципліни.** `relax` (послаблення без випливання) гірше за `off` на 2.96 % і гірше у 8 seed з 8 (t = −7.57, це не шум). `dive` — те саме послаблення, але з обмеженою тривалістю, примусовим випливанням і відкатом — гірше лише на 0.34 %. Тобто **фаза випливання відіграє 2.6 процентних пункти з 3.0**. Ваша початкова інтуїція, що без «випливання» механізм не працює, підтверджена числом.

Але сам механізм на Hanoi користі не дає: `dive` не краще за `off` (t = −1.92 при n = 8 — «не краще» обґрунтовано, «значимо гірше» ні).

**Чому — конкретна причина, а не «не спрацювало».** `improved = 0` при 100 зануреннях означає, що випливання повертається в той самий басейн. `_surface()` бере **найдешевше** рішення з пулу, а найдешевше — це найглибше занурене, тобто найближче до того самого локального оптимуму; `heal_network` потім піднімає його вгору **тим самим домінантним шляхом**, яким воно спускалося. Занурення проходить, але нікуди не приводить.

Три речі, які варто спробувати, перш ніж закривати тему:

1. **Випливати з найдальшого, а не з найдешевшого.** Брати рішення з пулу з максимальною відстанню Хеммінга від точки занурення. Це прямо адресує причину з `improved = 0`. Змінити треба один рядок у `_surface()`.
2. **Balerma.** Hanoi має 34 труби і 3 контури — допустима область тут, найімовірніше, не розділена на ізольовані басейни, тобто механізм адресує структуру простору, якої в Hanoi немає. 454 труби Balerma — саме той випадок, для якого ідея й придумана.
3. **Довші й глибші занурення** разом із пунктом 1.

**Рекомендація:** не тягнути це в основну гілку зараз. Механізм параметризований, інструментований і вимкнений за замовчуванням, тому він не заважає; повернутися до нього варто після етапу 7, коли `AcceptancePolicy` буде виділена — поступка належить саме їй.

**Для дисертації це готовий результат, і негативний результат тут не менш цінний:** ablation на трьох режимах з інструментацією, яка показує *чому* механізм не платить, плюс виміряний внесок фази випливання. Це сильніше, ніж «спробували, не вийшло».

Тести: 64 → **83** (`tests/test_dive.py` — 19 тестів на політику, цикл, облік заслуги й поступку в `ctx.score`; серед них перевірка, що `off` не послаблює нічого навіть при зіпсованому стані).

---

## 16. SUBMARINE: доопрацювання і рішення

Механізм доведений до кінця за трьома пунктами §15.5. Рішення — **вивести з основної гілки**.

### 16.1 Пункт 1: випливати з найдальшого рішення

Додано `surface_from` ∈ {`cheapest`, `farthest`} і `surface_attempts` (скільки кандидатів пробувати підняти за одне випливання). `farthest` обирає рішення пулу з максимальною відстанню Хеммінга від точки занурення — саме те, що адресує причину `improved = 0`.

Не допомогло: `improved` лишився 0.

### 16.2 Пункт 3: рідше і коротше

Виявилась справжня причина, і вона глибша за вибір кандидата. Замір на прогоні Hanoi (seed 42, глибина 2 м, 15 раундів):

```
кіків усього: 479, з них під зануренням: 267 (56%)
рекордів під зануренням: 0, поза зануренням: 6
```

**56 % бюджету пошуку проходить у режимі, в якому рекорд неможливий за побудовою** — шлях до рекорду вимагає `deficit == 0`. А недопустимим цей режим є не випадково: `heal_network` і `gradient_squeeze` під поступкою лікують і поліруюють до `h_min − ε`, тобто пошук збігається до оптимуму **послабленої** задачі, а він для початкової задачі строго недопустимий. Випливання потім просто піднімає рішення назад тим самим домінантним шляхом.

Тобто занурення не «дає пройти глибше» — воно переводить пошук у іншу задачу й забирає в нього бюджет.

Конфігурація «рідко і коротко» (3 м, 5 раундів, кулдаун 60, `farthest`, 3 спроби) знімає шкоду, бо занурення займає ~3 % раундів замість 56 %. Але разом зі шкодою зникає й будь-який вплив.

### 16.3 Пункт 2: Balerma — механізм там структурно неперевірний

На Balerma (454 труби) при 60 k симуляцій занурення **не спрацювало жодного разу**: воркер встигає пройти лічені раунди (884 сим/с, лог із 8 рядків, 0 кіків), тобто пошук навіть не доходить до режиму застою, за яким триґериться занурення.

Щоб перевірити механізм на Balerma, потрібен повний бюджет — 15 M симуляцій, це ≈ 4.7 години в один потік або ≈ година на 16 ядрах. У цьому середовищі це не вміщується, тому питання лишається відкритим саме для Balerma, і закрити його можна одним прогоном на вашій машині:

```bash
# два прогони, критерій рішення — чи буде improved > 0
python main.py --inp InputData/Balerma/Balerma.inp --costs InputData/Balerma/costs.csv \
               --run_mode analytical --units mm --hmin 20 --cores 16 --max_sims 15000000 --seed 42 \
               --dive dive --dive_depth 2.0 --dive_rounds 15 --dive_surface farthest
grep "DIVE STATS" OutputDataExperiments/<папка>/logs/worker_*.txt
```

Якщо `improved = 0` і там — механізм можна закривати остаточно.

### 16.4 Підсумковий замір на Hanoi

| конфігурація | mean | stdev | проти `off` | t |
|---|---|---|---|---|
| `off` (без механізму) | **6.1023** | 0.0124 | — | — |
| `relax`: стала поступка 2 м | 6.2831 | 0.0576 | −2.96 % | −7.57 |
| `dive`: 2 м, 15 раундів, `cheapest` | 6.1231 | 0.0302 | −0.34 % | −1.92 |
| `dive`: 3 м, 5 раундів, `farthest`×3, кулдаун 60 | 6.1069 | 0.0276 | −0.07 % | −0.43 |

**124 занурення в п'яти параметризаціях** (глибина 0.5 / 2 / 3 / 10 м, тривалість 5 / 8 / 15 раундів, `cheapest` / `farthest`, 1 / 3 спроби) — **`improved = 0` у кожній**. Жодного разу випливання не побило рекорд, актуальний на момент випливання.

### 16.5 Рішення: вивести з основної гілки

Механізм доведений до робочого стану, інструментований і зміряний. Користі на Hanoi він не дає ні в одній з п'яти параметризацій; найкраще, чого вдалося досягти, — зробити його нешкідливим, звівши до ~3 % раундів.

Тримати ~200 рядків непрацюючої машинерії в проєкті, головною скаргою на який була саме надлишковість коду, суперечить сенсу рефакторингу. Тому:

```bash
# зафіксувати роботу й позначити її, щоб не втратилась
git add -A
git commit -m "SUBMARINE: strategic oscillation, fully instrumented; no gain on Hanoi (124 dives, improved=0)"
git tag experiment/submarine

# прибрати з гілки рефакторингу
git revert --no-edit HEAD
```

Після цього код механізму лишається доступним за тегом (`git show experiment/submarine`), результат — у цьому звіті, а гілка рефакторингу повертається точно до стану етапу 3.

**Що варто зберегти попри це** — узагальнення, які виникли по дорозі й не залежать від механізму: `SimStats`, `ctx.score` як єдина точка, `--sequential`, `--seed`, фіксація `PYTHONHASHSEED`, `--time_limit`. Вони вже в попередніх коммітах і revert їх не торкається.

**Альтернатива, якщо код потрібен поруч із текстом дисертації:** не робити revert, а лишити механізм вимкненим (`--dive off` за замовчуванням, режим `off` перевірено на побітову ідентичність). Ціна — ~200 рядків і три параметри в `main.py`, які нічого не роблять у типовому прогоні. Обирати вам; я рекомендую revert із тегом, бо результат зберігається в звіті й у git, а код лишається чистим.

### 16.6 Що з цього лишається для дисертації

Це повноцінний негативний результат із поясненням механізму, а не «спробували, не вийшло»:

1. **Виміряна ціна дисципліни.** Стала поступка без випливання — −2.96 % і гірше у 8 seed з 8 (t = −7.57). Той самий допуск із обмеженою тривалістю, примусовим випливанням і відкатом — −0.34 %. Фаза випливання відіграє 2.6 процентних пункти з 3.0. Це кількісне підтвердження вихідної інтуїції, що без «випливання» механізм не працює.
2. **Причина, чому не працює й з випливанням.** Під поступкою пошук збігається до оптимуму послабленої задачі, який для початкової строго недопустимий, і забирає до 56 % раундів у режимі, де рекорд неможливий. Це не дефект реалізації, а властивість самої схеми при такій інтеграції.
3. **Дві методичні пастки**, кожна з яких окремо здатна дати хибний висновок: облік заслуги від точки занурення замість поточного рекорду (завищує ефект) і проведення поступки лише в гейти прийняття, а не в конструктори рішень (робить механізм інертним, причому непомітно — числа виглядають правдоподібно).
4. **Відкрите питання:** великі мережі. На Balerma механізм при доступному тут бюджеті не встигає спрацювати; команда для перевірки — §16.3.

---

## 17. Етап 4: один модуль запуску воркера

### 17.1 Що було

Три майже однакові реалізації:

| Розташування | Рядків |
|---|---|
| `AnalyticalSolver.worker_task` (classmethod) | 46 |
| `analytical_worker_task` у `main.py` | 82 |
| `analytical_worker_task` у `gui.py` | 82, побайтово ті самі |

Обидві точки входу на старті робили `AnalyticalSolver.worker_task = staticmethod(analytical_worker_task)`, тобто classmethod у пакеті існував лише щоб бути перезаписаним. А класи солвера діставалися так:

```python
solver_module = sys.modules[AnalyticalSolver.__module__]
SolverContext = solver_module.SolverContext
LocalSearch   = solver_module.LocalSearch
KickStrategies = solver_module.KickStrategies
IslandWorker  = solver_module.IslandWorker
```

— через приватний простір імен чужого модуля. Працювало лише тому, що `orchestrator.py` випадково мав ці імена; будь-яка зміна його імпортів зламала б GUI без помилки на етапі імпорту.

Симулятор процесу зберігався як `worker_sim_instance` у `__main__`, і кожна точка входу оголошувала цей глобал сама.

### 17.2 Що стало

`analytical_solver/worker.py` (199 рядків) — єдина реалізація:

* `WorkerTask` (frozen dataclass) замість позиційного кортежу на 12 елементів із двома `if len(args) > N` на хвості;
* `WorkerResult` (NamedTuple) замість кортежу з порожнім третім елементом, який усі розпаковували в `_`;
* `init_worker_process(inp_file, config)` — initializer пулу, володіє симулятором процесу;
* `set_simulator(sim)` — явна передача для запуску в поточному процесі (`--sequential`);
* `run_island_worker(task)` — один шлях і для `pool.apply_async`, і для послідовного виклику;
* `_WorkerLogger` — раніше вкладений клас, дубльований у двох файлах.

Мавпячий патч, classmethod `worker_task`, обидві копії `analytical_worker_task` і магія `sys.modules[...]` видалені. Симулятор більше не живе в `__main__`.

Заодно прибрано дві дрібніші дублікації: `get_temp_root()` (був у `main.py` і `gui.py` ідентично) переїхав у `worker.py`, а класифікація мережі — `classify_network()` і `beam_width_for()` у `context.py` замість чотирьох копій ланцюга `"SMALL" if n < 50 else (...)` (у `worker_task`, `AnalyticalSolver.__init__`, `main.py`, `gui.py`).

Знайдено мертвий атрибут: `ctx.log_file` присвоювався у трьох місцях і ніде не читався (`ctx.log` просто друкує в stdout, який воркер перенаправляє у файл). Видалений.

### 17.3 Перевірка

| | було | стало |
|---|---|---|
| `main.py` | 318 | **254** |
| `gui.py` | 1331 | **1221** |
| `analytical_solver/orchestrator.py` | 1607 | **1502** |
| `analytical_solver/worker.py` | — | 199 |
| реалізацій запуску воркера | 3 | **1** |
| копій класифікації мережі | 4 | **1** |
| тестів | 64 | **84** |

Еталон Hanoi (`--sequential`) збігся **точно**: seed 1 → 6.1090 M$, seed 42 → 6.1182 M$. Паралельний режим перевірено окремо — пул піднімається через `init_worker_process`, логи всіх трьох воркерів пишуться, помилок немає. `fast_analytical` → 6.8013 M$, без змін.

Один із наявних тестів довелося оновити, і це показово: `tests/test_solver_regression.py` підсовував симулятор у `__main__.worker_sim_instance`, бо саме звідти його діставав старий `worker_task`. Тепер він викликає `set_simulator(sim)` — тест перестав імітувати внутрішню магію.

`tests/test_worker.py` (20 тестів) фіксує новий контракт: межі класифікації мережі дослівно як були, незмінність `WorkerTask`, окремий `global_archive` на кожне завдання (пастка `default_factory`), відсутність поля-заглушки в `WorkerResult`, і зрозуміла помилка при запуску без симулятора.

### 17.4 Що лишилось у `solve_standalone`

CC усе ще 60. Метод тримає дві речі, які належать різним рівням: розсилку завдань воркерам і 60-рядковий монітор живого прогресу з `time.sleep(1.0)`, який друкує статус і оновлює спільний архів. Монітор — це презентація, а не алгоритм; він переїде в `runner.py` на етапі 5 разом із дублюванням між `main()` і `run_optimization()`.

---

## 18. Етап 5: один сценарій запуску для CLI і GUI

Це найбільший залишок дублювання в проєкті і перший етап, який суттєво чіпає `gui.py`.

### 18.1 Що дублювалось

`main()` у `main.py` (154 рядки) і `PipelineOptimizerApp.run_optimization()` у `gui.py` (171 рядок) робили той самий сценарій: збирали `GAConfig`, підіймали `WaterSimulator`, створювали пул, крутили цикл `runs`, перетворювали діаметри назад в індекси, вибирали чемпіона, експортували звіти. Не побайтово — і саме тому це дублювання було небезпечним: дві копії вже **розійшлися**.

* при `runs > 1` CLI зберігав рішення кожного прогону **без історії збіжності та без графіків**, GUI — з ними;
* `clean_temp_root()` існував у двох версіях: у `gui.py` — з циклом спроб і паузою (на Windows EPANET Toolkit тримає дескриптори `.rpt`/`.bin` після завершення воркера), у `main.py` — спрощена на 9 рядків, яка на Windows лишала сміття;
* тільки GUI тримав `self.current_pool` і знав, що пул треба гасити при «ЗУПИНИТИ».

Розбіжність №1 вирішена свідомо на користь GUI-поведінки: тепер обидві точки входу зберігають повні звіти кожного прогону. Розбіжність №2 — на користь версії з `gui.py`.

### 18.2 Що з'явилось

`runner.py` (356 рядків), три предметні поняття замість одного довгого сценарію:

* **`RunRequest`** — усе, що потрібно знати, щоб провести серію прогонів. Похідні величини стали властивостями, а не розкиданими виразами: `effective_cores` (0 = усі ядра, `--sequential` = 1) і `solver_time_limit` (`None` = вивести з калібрування, `0` або менше = зупинка лише за бюджетом симуляцій, `--sequential` без явного `--time_limit` = `inf`).
* **`RunRequest.from_text_fields`** — толерантний розбір текстових полів GUI (`try: int(...) except: 0`). Виділений саме для того, щоб його можна було перевірити тестами: `tkinter` у контейнері немає, і поки ця логіка жила всередині `run_optimization`, вона була недосяжною для тестів **у принципі**.
* **`RunResult`** — результат прогону з `total_evals`, `status` і `as_summary_row()` замість трьох паралельних списків (`all_results`, `all_histories`, `all_individuals`), які індексувались вручну.
* **`OptimizationRunner`** — `prepare()` / `run(base_dir)` / `export(results, base_dir)`. Три необов'язкові хуки — і це рівно те, чим GUI відрізняється від CLI: `on_progress` (дашборд), `should_stop` (кнопка «ЗУПИНИТИ»), `on_run_start` (очистити графік збіжності).

`prepare()` заодно додав **pre-flight**: один розрахунок у головному процесі перед підняттям пулу. Раніше зламаний `.inp` або неузгоджена таблиця цін валили воркер, і в лог летіло `WORKER INIT CRASH` замість зрозумілої помилки.

### 18.3 Метрики

| | було | стало |
|---|---|---|
| `main.py` | 254 | **119** |
| `gui.py` | 1221 | **1096** |
| `runner.py` | — | 356 |
| копій сценарію запуску | 2 (розійшлися) | **1** |
| копій `clean_temp_root` | 2 (різні) | **1** |
| тестів | 84 | **126** |

`main()` тепер робить рівно те, що мусить робити точка входу CLI: розбір аргументів → фіксація `PYTHONHASHSEED` і seed → лог → `RunRequest` → `prepare/run/export`. `run_optimization()` у GUI — те саме, але параметри з полів вводу, а прогрес у дашборд.

### 18.4 Перевірка

Прогони Hanoi у контейнері (`--units in`, `analytical`):

| Команда | Результат | Симуляції |
|---|---|---|
| `--sequential --seed 42` | 6.0812 M$, P=30.01 м | 1 001 158 |
| `--sequential --seed 42` (повтор) | 6.0812 M$, P=30.01 м | 1 001 158 |
| `--cores 1 --seed 42` | 6.0812 M$, P=30.01 м | 1 001 296 |
| `--cores 3 --runs 2 --max_sims 120000 --time_limit 0 --seed 5` | 6.0974 M$ обидва прогони | 122 731 / 122 769 |

Повтор дав **той самий вектор рішення** (`solution_champion.csv` збігся за md5) і ту саму кількість симуляцій — відтворюваність, встановлена на етапі 0, збережена. `--cores 1` дав той самий вектор, що й `--sequential`, тобто обидва шляхи узгоджені. Паралельний прогін на 2 runs перевірив те, чого не перевіряє жоден однопрогінний: пул через `init_worker_process`, логи трьох воркерів, `_export_one` для кожного прогону, `runs_summary.csv` і звіти чемпіона — повне дерево з 25 файлів на місці.

**Обмеження цієї перевірки, яке варто назвати прямо.** Числа 6.1090 / 6.1182 у таблиці §0 і §17.3 отримані на вашій машині, а `6.0812` — у контейнері. Точка зупинки за замовчуванням виводиться з калібрування швидкості симулятора (§12.4), тож між машинами вона не однакова, і порівнювати ці числа між собою некоректно. Тому **порівняння «до / після етапу 5» мусить зробити один прогін на вашій машині**:

```bash
python main.py --units in --cores 1 --seed 42
```

Має дати ваші звичні 6.1182 M$. Якщо число інше — це регресія етапу 5, і її треба шукати в аргументах, з якими створюється `AnalyticalSolver`.

Саме тому аргументи солвера зафіксовані не прогоном, а тестами (`tests/test_runner.py`, розділ «контракт із солвером»): `_SpySolver` перехоплює конструктор і перевіряє `n_workers`, `max_sims`, `time_limit_sec`, `pool`, `log_dir`, `v_opt` для послідовного, паралельного і дефолтного режимів, а також що `fast_analytical` йде в `solve_fast`, а `analytical` — у `solve_standalone`. Стохастичний прогін на Hanoi таку зміну виявив би лише випадково; тест ловить її за 1 с.

`tests/test_runner.py` — 42 тести: дефолти й валідація `RunRequest`, толерантний розбір полів (`""`, `"   "`, `"abc"`, `None`, `"0"`, `"-4"`), `effective_cores`, три режими `solver_time_limit`, `RunResult`, вибір чемпіона (допустимі перед дешевими) і зворотне відображення діаметрів разом із гілкою `ValueError -> останній індекс`.

`gui.py` перевірено `ruff` і `py_compile`: `tkinter` у контейнері недоступний, тому імпорт-тест неможливий. Це і є причина, чому розбір полів винесено в `RunRequest.from_text_fields` — тепер єдине, що лишилось у GUI неперевіреним, це зчитування рядків із віджетів.

### 18.5 Побічна знахідка: неузгодженість одиниць не діагностується

Знайдено під час перевірки, коли я сам запустив Hanoi без `--units in`. `InputData/Hanoi/costs.csv` задає діаметри в **дюймах** (12, 16, …, 40), а дефолт `--units` — `mm`. При розбіжності `12.0` читається як 12 мм, діаметри виходять у 25 разів меншими, тиски — порядку −10⁸ м, і прогін:

* виводить `[WARNING] No valid seeds received. Injecting emergency max-diameter seed.`;
* спалює **весь** бюджет у 1 000 000 симуляцій (9.7 хв) у циклі `[EMERGENCY] Beam search deadlocked` — 418 093 однакових рядків, 89 МБ лога воркера;
* завершується `FINAL RESULT: 10.9698M$ | P=-350892532.03m | FAIL` і **записує звіти** так само, як успішний прогін.

Тобто типова помилка у прапорці коштує десять хвилин і не супроводжується жодним повідомленням про причину. Два дешевих запобіжники (обидва — у `prepare()`, поруч із наявним pre-flight):

1. якщо рішення «усі максимальні діаметри» недопустиме за тиском — зупинитись із повідомленням, що задача нерозв'язна за цієї таблиці діаметрів / одиниць. Це коректний критерій: якщо найбільші труби не тримають тиск, жодне рішення не триматиме;
2. порівняти медіану діаметрів з таблиці з медіаною діаметрів у самому `.inp` і попередити при розбіжності на порядок — `.inp` уже прочитаний через `wntr`, тому це кілька рядків.

Друге — попередження, перше — зупинка. Обидва варті окремої дрібної правки; в етап 5 не входять, бо змінюють поведінку запуску.

### 18.6 Що НЕ переїхало

Монітор живого прогресу з `solve_standalone` (60 рядків, `time.sleep(1.0)`, друк статусу й оновлення спільного архіву) лишився в `orchestrator.py`. У §17.4 я планував перенести його в `runner.py` на цьому етапі, і від цього свідомо відмовився: монітор не просто друкує, він **виконує міграцію** — читає `shared_progress` і кладе в нього оновлений глобальний архів. Це частина алгоритму острівної моделі, а не презентація, і виносити її в раннер означало б розірвати асинхронну синхронізацію між точкою, де архів оновлюється, і точкою, де він розсилається. `solve_standalone` розбирається на етапі 7 разом із `StrategySelector` / `AcceptancePolicy`.

---

## 19. Етап 6: `netdraw.py` — одна реалізація малювання мережі

### 19.1 Що дублювалось

Карта мережі малювалась у **чотирьох** місцях:

| Місце | Звідки діаметри |
|---|---|
| `plot.py::plot_network_map` | вектор рішення + таблиця типорозмірів |
| `gui.py::_update_live_network_graph` | живе рішення під час пошуку |
| `gui.py::_draw_basic_topology` | діаметри з самого `.inp` |
| `gui.py` (перегляд результатів) | `solution.csv` з архіву прогону |

Різне в них — лише останній стовпець. Спільне — усе решта: побудова графа й координат, дискретна шкала `turbo` по унікальних типорозмірах, формула товщини `1 + 4 * (d - min) / (max - min + 1e-6)`, легенда з підписами `f"{d:g}"`, розміри й кольори вузлів. Формула товщини скопійована чотири рази дослівно, включно з `1e-6`; `#95a5a6`, `#3498db`, `#2ecc71` і `node_size=15` — так само.

Копії вже розійшлися, і одна з розбіжностей — дефект:

* **координати.** `plot.py` збирає їх по вузлах і має запасний `kamada_kawai_layout` для мереж без координат; усі три копії в GUI викликали `query_node_attribute('coordinates')` без запасного варіанту. На мережі без координат малювання в GUI падало, і вікно тихо показувало готовий PNG замість інтерактивного графа — помилка не діагностувалась, бо її ловив `except`.
* **фільтр `if n in G.nodes`** для вузлів був лише в одній копії з чотирьох.
* **порядок значень.** Вектор рішення індексується порядком `wn.pipe_name_list`, а `networkx` малює ребра у своєму порядку. `plot.py` розв'язував це власним `edgelist`, три копії в GUI — відображенням через імена труб. Два різні способи розв'язати ту саму невідповідність.

### 19.2 Що з'явилось

`netdraw.py` (334 рядки), чотири поняття:

* **`NetworkGeometry`** — граф, координати (із запасним layout для всіх викликів), порядок труб, групи вузлів. `wntr` імпортується локально в `load`, тому решта модуля тестується без EPANET.
* **`DiameterScale`** — дискретна шкала: `color_indices`, `widths`, `unique`, `has_data` і `attach_colorbar(fig, ax, label, existing=...)`. `existing` — саме те, що потрібно живій топології: вона оновлює легенду, а не створює нову (інакше вони накопичувались би збоку фігури).
* **`draw_nodes` / `draw_edges` / `draw`** — малювання; `draw` повертає шкалу для легенди.
* **`diameters_from_indices` / `diameters_from_solution` / `diameters_from_inp` / `diameters_from_table`** — чотири джерела значень, по одному на кожен колишній виклик. **Усі повертають список у порядку `geometry.edge_pipe_names`** — це і є той єдиний контракт, якого раніше не було.

Одна невідповідність усунена свідомо: `plot.py` більше не передає власний `edgelist`, тобто PNG-звіт малює ребра в тому ж порядку, що й GUI. Ціна — 11 пікселів із 17 640 000 (див. нижче), виграш — один порядок замість двох.

### 19.3 Метрики

| | було | стало |
|---|---|---|
| `gui.py` | 1096 | **952** |
| `plot.py` | 318 | **271** |
| `netdraw.py` | — | 334 |
| копій малювання мережі | 4 | **1** |
| копій формули товщини | 4 | **1** |
| тестів | 126 | **149** |

`gui.py` від початку рефакторингу: 1331 → **952** рядки (−28 %).

### 19.4 Перевірка

**PNG-звіт — попіксельне порівняння.** Функція `plot_network_map` витягнута з немодифікованої копії проєкту через AST і виконана поруч із новою на тих самих вхідних даних:

| Випадок | Розбіжність |
|---|---|
| випадкове рішення (34 труби, 6 типорозмірів) | **11 пікселів із 17 640 000** |
| однакові діаметри в усіх трубах (перевірка `+1e-6`) | **0 пікселів** |

11 пікселів — це пляма 3×7 в одній точці, де дві труби перетинаються і змінився порядок їх малювання. Це вся ціна уніфікації порядку; заголовок, легенда, кольори, товщини й вузли збігаються побайтово.

**Три шляхи GUI** перевірені без tkinter — `netdraw` від нього не залежить:

* початкова топологія з реального `Hanoi.inp` — шкала побудована, легенда створена;
* жива топологія — два кадри поспіль, легенда **той самий об'єкт**, підписи оновились з `[16]` на `[12, 40]`;
* перегляд результатів — реальний `solution_champion.csv` прочитаний, стовпець `Diameter (inch)` знайдений за назвою, всі 34 труби зіставлені, жодної невідомої.

**`tests/test_netdraw.py`** — 23 тести. Найцінніші три фіксують саме те, на чому копії розходились: `diameters_from_*` перевіряються на фейковій геометрії, де труби в графі навмисно **не** в порядку `pipe_name_list` (`P3, P1, P2`), тому тест падає, якщо хтось знову зіставить значення за позицією; окремо перевіряється, що вузол зі списку, якого немає в графі, не потрапляє в малювання; і що однакові діаметри не дають ділення на нуль. Плюс порядок z-шарів в обох варіантах (`nodes_on_top`) — читається з `ax.collections`.

`ruff` чистий (лишається відомий `kicks.py:12 ARG002`), `py_compile` проходить, 149 тестів за 21 с.

### 19.5 Побічна знахідка: `Hanoi.inp` не містить реальних діаметрів

Помічено при перевірці початкової топології: у `[PIPES]` файлу Hanoi стоять діаметри-заглушки — `20000` і `0.0001` мм. Це нормально для задачі оптимізації (діаметри й є те, що шукається), але вкладка «початкова топологія» показує для Hanoi шкалу `[0.0001, 20]` — тобто безглузду картинку. Так було й до рефакторингу; це питання до вхідних даних, не до коду, і в етап 6 не входить. Варіант на майбутнє: якщо всі діаметри в `.inp` однакові або виходять за фізичні межі, малювати мережу сірим і підписати «діаметри не задані» замість легенди.

---

## 20. Етап 7: план розбору `_apply_kick` (до виконання)

`_apply_kick` — 314 рядків, CC **149**, найскладніший метод проєкту. Це не «довгий метод», а шість різних рівнів рішення в одному тілі:

| Рівень | Рядків | Що робить |
|---|---|---|
| температура | ~6 | `real_T`, `T`, `is_late_game`, вибір гарячої/холодної таблиці |
| вибір стратегії | ~45 | untried → VND-random → UCB1, штраф за серію, фільтр за поспіль-невдачами |
| вибір цілі кіка | ~10 | три гілки за `real_T` (найдальший / випадковий / за `round_idx % 3`) |
| виклик кіка | ~25 | реєстр + 13 kwargs + `try/except` (зроблено на етапі 3) |
| ремонт і поглиблення | ~45 | `heal_network` при недоборі тиску, швидкий і глибокий `gradient_squeeze`, заморозка консенсусу архіву |
| прийняття | ~130 | п'ять взаємовиключних гілок, і в кожній — ще й нагорода бандита, оновлення JADE, табу, лог |

Дублювання всередині методу, яке видно в цифрах:

* **оновлення JADE `mu_*` — у 6 місцях**, кожне повторює ланцюг `if strategy == "RUIN_RECREATE" / elif "SMART_PERTURB" / elif "SPATIAL_PERTURB"`, з чотирма різними альфами (`0.15` адаптивна, `0.03 * proximity`, `× 0.90`, `× 0.95`) і різними підлогами (`0.02` / `0.03` / `0.05`);
* **зміна ваг бандита — у 7 місцях** (`× 0.25` при невдачі, `× 0.50` при вибуху вартості, `+ reward` на рекорді, `× 0.80` для решти, `+ soft_reward` при прийнятті в пул, `+ reward` у hail-mary, `× 0.3` при флеші);
* **`active_wins` / `active_tries`** — псевдоніми, які на початку методу вказують то на гарячу, то на холодну таблицю; через це чотири словники стану читаються в коді як два.

### 20.1 Три поняття

**`StrategySelector`** — власник `strat_wins_hot/cold`, `strat_tries_hot/cold`, `strat_consecutive_fails`, `_last_strategy`, `_strategy_burst_count`.

```
pick(temperature, pool) -> Choice(strategy, mode_label, bandit_active)
note_failure(strategy) / note_success(strategy)
reward(strategy, amount) / penalise(strategy, factor)
decay_all(factor)
```

Гаряча/холодна таблиця обирається всередині за `is_late_game`, тому псевдонімів `active_*` більше не існує. `bandit_active` повертається один раз замість шести перевірок `if use_bandit`.

**`AdaptiveRates`** (JADE) — власник `mu_ruin_pct`, `mu_perturb_pct`, `mu_spatial_pct`, `mu_escape_pct` і **однієї** таблиці відповідності:

```
RATES = {"RUIN_RECREATE": ("ruin", 0.02), "SMART_PERTURB": ("perturb", 0.03),
         "SPATIAL_PERTURB": ("spatial", 0.05)}

as_kick_args() -> dict            # чотири mu_* у kwargs кіка
learn(strategy, used_pct, alpha)  # EMA
shrink(strategy, factor)          # з підлогою з таблиці
learn_escape(used_pct)
```

Шість копій ланцюга if/elif стають шістьма викликами.

**`AcceptancePolicy`** — п'ять гілок як **рішення**, а не як дія:

```
classify(...) -> Verdict(kind, cost, score, solution)
kind ∈ {RECORD, ESCAPE, POOL, HAIL_MARY, REJECT}
```

Порогові вирази отримують імена: `explosion_threshold` (`best × (1.5 + 0.5·T)`), `pool_window` (`best × (1.03 + 0.07·T)`), `max_pool_gap`, `hb_margin`, `deep_passes_for(gap, T)`. `_apply_kick` після цього читається як шість кроків: вибрати стратегію → вибрати ціль → кікнути → відремонтувати → класифікувати → застосувати вердикт.

### 20.2 Головний ризик і як його зняти

`_apply_kick` мутує ~20 полів воркера, і **порядок мутацій значущий**. Найгостріший приклад:

```python
self.run_best_cost, self.run_best_sol = effective_cost, final_sol
self.stagnation_counter = 0 if diff > (self.run_best_cost * 0.005) else ...
```

Порог `0.005` рахується від **уже нового** `run_best_cost`, не від старого. Будь-яке переставляння цих двох рядків тихо змінює поведінку. Тому план такий: **вердикт виноситься, застосування вердикту лишається в тілі методу**, дослівно в тому ж порядку. Спершу — характеризаційні тести на послідовність мутацій (кожна з п'яти гілок: що саме змінилось у полях воркера після одного кіка), і лише потім рух коду.

### 20.3 Два дефекти, знайдені при аналізі

**Пастка `p_max` вижила у двох місцях.** Етап 2 прибрав змінну `feas`, у яку розпаковувався `p_max`, у 22 місцях — але я шукав за іменем `feas`, а тут вона зветься інакше:

```python
quick_cost, quick_p, quick_f, _ = self.ctx.get_cached_stats(quick_sol)   # рядок 799
if quick_f and quick_p >= base_h_min and ...

q_c, q_p, q_f, _ = self.ctx.get_cached_stats(quick)                      # рядок 935
if q_f and q_p >= base_h_min and q_c < self.run_best_cost:
```

`quick_f` і `q_f` — це `p_max`, а не прапорець допустимості. Наслідків немає (доказ той самий, що в §0: при `h_min > 0` умова `p_min >= h_min` уже все вирішує, а при збої симуляції `p_min = -1` відсікає результат), але це той самий клас пастки, і мій власний свіп етапу 2 його пропустив. Виправляється разом із розбором.

**`ABLATION_MODE = False`** — локальна константа, зашита в тіло методу, яка гасить шість гілок: фіксує `T = 0.5`, знімає послаблення `h_min` і вимикає всі оновлення JADE. Зараз ці гілки недосяжні, тобто це заглушка в термінах початкової ревізії. Пропоную те саме рішення, що на етапі 1 для `classical_descent`: зробити параметром (`IslandWorker(..., ablation=False)`), щоб ablation-прогін не вимагав правки вихідника. Видаляти не пропоную — ablation-порівняння знадобиться для дисертації.

### 20.4 Порядок робіт

1. характеризаційні тести на п'ять гілок прийняття (фіксують послідовність мутацій);
2. `AdaptiveRates` — найбезпечніше, шість копій → одна таблиця;
3. `StrategySelector` — прибирає псевдоніми `active_*` і шість `if use_bandit`;
4. `AcceptancePolicy` — вердикт; застосування лишається на місці;
5. `ablation` як параметр; виправлення `quick_f` / `q_f`;
6. еталон Hanoi (`--units in --cores 1 --seed 42`) мусить дати **6.0812 M$ при 1 001 296 симуляціях** — після цього етапу це вже перевірено побітово на двох машинах, тому будь-яке розходження означає регресію.

### 20.5 Крок 0–1 зроблено: воркер став валідним, сітка готова

**Знайдено при спробі викликати `_apply_kick`:** він падав з `AttributeError: 'IslandWorker' object has no attribute 'base_dyn_bonus'`. AST-аналіз класу дав повну картину — сім полів присвоювались **лише в `run()`**, хоча читаються з інших методів:

| Поле | Читається в | Присвоювалось у |
|---|---|---|
| `base_dyn_bonus` | **9 методах** | тільки `run()` |
| `stag_limit` | **5 методах** | тільки `run()` |
| `reserve_pool` | `_emergency_pool_diversity` | тільки `run()` |
| `_last_ping_sims`, `_last_ping_time`, `_migration_interval_sims`, `_last_migration_sims` | `run()` | тільки `run()` |

Тобто `IslandWorker` після `__init__` був у невалідному стані: будь-який його метод, викликаний до `run()`, падав у глибині кіка. Це той самий клас дефекту, що «поля GUI» на етапі 2 (§0, дефект 3), і він робив метод нетестовним у принципі.

Виправлено: усі сім оголошені в `__init__`. Поведінка не змінюється — `run()` перераховує `base_dyn_bonus` і `stag_limit` перед першим кіком і далі щороунду, `reserve_pool` будується до циклу, тому початкові значення до пошуку не доходять. Вони існують, щоб об'єкт був валідним з моменту створення.

**Одна мікро-екстракція** зроблена одразу, бо без неї найтонше правило методу неможливо зафіксувати:

```python
RECORD_RESET_FRACTION = 0.005

@staticmethod
def _stagnation_after_record(diff, new_best_cost, current_counter):
    if diff > new_best_cost * IslandWorker.RECORD_RESET_FRACTION:
        return 0
    return max(0, current_counter - 2)
```

Правило застосовується у двох місцях (рекорд і hail-mary-рекорд) і в обох рахує порог від **нового** рекорду. Тест `test_stagnation_threshold_is_measured_against_the_new_best_not_the_old` перевіряє саме це: бере `diff` у вузькому проміжку між двома порогами (для рекорду 1 млн це 5000 … 5020) і фіксує, що трактування — «суттєвий виграш». Прогоном цю різницю в 0.5 % не видно взагалі.

**`tests/test_apply_kick.py` — 28 тестів.** Стратегія форсується підміною `pool_for` на пул з однієї стратегії: тоді її обирає і UCB1-гілка, і VND-гілка, вибір перестає бути випадковим, а решта методу працює без змін. Температура задається через `stagnation_counter / (stag_limit * 3)`, гілка прийняття — через зсув `run_best_cost` відносно вартості базового рішення.

Усі **шість** розв'язок методу досяжні детерміновано, і кожна закріплена параметром у тесті:

| Розв'язка | Як досягається |
|---|---|
| RECORD | `SHOCK`, застій 1, рекорд × 1.0 |
| ESCAPE | `BASIN_ESCAPE`, застій 5, рекорд × 0.6, архів із 3 рішень |
| POOL | `SHOCK`, застій 12 (T = 1.0), рекорд × 0.6 |
| HAIL MARY → рекорд | `SHOCK`, застій 12, рекорд × 0.55 |
| HAIL MARY → прийняття | `SMART_PERTURB`, застій 12, рекорд × 0.32 |
| REJECT | `SHOCK`, застій 1, рекорд × 0.30 |
| відкид за вибухом вартості | `SHOCK`, застій 12, рекорд × 0.30 |
| невдача самого кіка | `BASIN_ESCAPE` з порожнім архівом |

Окремий параметризований тест `test_each_outcome_is_reachable` перевіряє саму досяжність: якщо після розбору якась гілка стане недосяжною, впаде він, а не десяток тестів навколо.

Зафіксовані, зокрема, ті переходи, які легко зламати непомітно: ESCAPE гасить обидві таблиці бандита на 70 % і очищає **чотири** різні пам'яті кіків, виставляє `_force_flush_next` і `ipc_immunity = 50`; HAIL-MARY-прийняття не скидає застій, а **виставляє** його в `1.8 × ліміту`; невдача кіка коштує стратегії 75 % ваги з підлогою 0.1; відкид за вибухом вартості не змінює **нічого**, крім статистики.

177 тестів, 21 с. Еталон Hanoi після цих правок — 6.0812 M$ / 1 001 296 симуляцій, без змін.

### 20.6 Крок 2–3: `AdaptiveRates` і `StrategySelector`

**`AdaptiveRates`** (122 рядки) — розміри кроку кіків і правила їх навчання. П'ять місць оновлення в `_apply_kick` стали п'ятьма викликами, а ланцюг `if strategy == "RUIN_RECREATE" / elif / elif` — однією таблицею:

```python
STEP_SIZES = {                      # стратегія: (ключ, старт, підлога)
    "RUIN_RECREATE":   ("ruin",    0.05, 0.02),
    "SMART_PERTURB":   ("perturb", 0.10, 0.03),
    "SPATIAL_PERTURB": ("spatial", 0.20, 0.05),
}
```

Чотири альфи отримали імена й місце: `record_alpha(used_pct)` (базова 0.15, послаблена логарифмом розміру кроку — великий крок міг дати виграш випадково, тому довіряється менше), `pool_alpha(pool_gap)` (0.03, згасає до нуля на межі вікна близькості), `ESCAPE_ALPHA` (0.10).

Головне, що стало видимим: **дві політики звуження не однакові**. Вибух вартості звужує крок на 10 % **без підлоги** (крок явно завеликий — межа не має його тримати), звичайний відкид звужує на 5 % **з підлогою**. У початковому коді це були два фрагменти в різних гілках, і різницю було видно лише при уважному читанні; тепер це два методи з іменами — `shrink_after_explosion` і `shrink_after_reject` — і два тести, які цю різницю фіксують (40 вибухів опускають крок нижче підлоги, 200 відкидів — точно до неї).

**`StrategySelector`** (203 рядки) — вибір стратегії і статистика. П'ять словників стану воркера (`strat_wins_hot/cold`, `strat_tries_hot/cold`, `strat_consecutive_fails`) плюс `_last_strategy` і `_strategy_burst_count` переїхали сюди, разом із 55 рядками вибору й **сімома** місцями зміни ваг.

`pick(pool, T, T_effective, is_late_game)` повертає `Choice(strategy, mode_label, bandit_active)`. Три наслідки:

* псевдонімів `active_wins` / `active_tries` більше не існує — фазу («холодну» чи «гарячу» таблицю) обирає сам селектор, і переплутати її ззовні неможливо;
* `if use_bandit` у шести місцях замінився одним полем результату;
* сім різних виразів зміни ваг стали чотирма методами з межами в іменах: `reward` (стеля 4.0), `penalise` (підлога 0.1 — нижча за звичайну 1.0, бо покарана стратегія мусить мати шанс повернутись, але не швидко), `decay_others`, `reset_after_escape`.

Формули нагород теж отримали імена: `record_reward(improvement)` = `1 + 100 × відносний виграш`, `diversity_reward(ratio, T)` = `0.5 × ratio × T`.

`ablation` уже є параметром конструктора селектора; локальна константа `ABLATION_MODE` у тілі `_apply_kick` знімається разом із `AcceptancePolicy` на наступному кроці, коли зникнуть останні гілки, які її читають.

### 20.7 Метрики кроків 2–3

| | було | стало |
|---|---|---|
| `_apply_kick`, CC | 149 | **106** |
| `_apply_kick`, рядків | 314 | **~200** |
| `analytical_solver/orchestrator.py` | 1536 | **1452** |
| копій ланцюга if/elif по стратегії (JADE) | 5 | **1** (таблиця) |
| місць зміни ваг бандита | 7 | **4 методи** |
| полів стану воркера під статистику стратегій | 7 | **1 об'єкт** |
| тестів | 177 | **225** |

**Перевірка.** Найцінніше тут — що **всі 28 характеризаційних тестів пройшли без жодної зміни в assertions**. Правилась лише функція `snapshot()` — одна точка, яка знає, звідки читати стан; сама поведінка в усіх шести розв'язках лишилась така, як зафіксовано до розбору. Плюс 24 нових тести на `StrategySelector` і 20 на `AdaptiveRates`, які перевіряють уже правильність, а не лише незмінність: незалежність гарячої й холодної таблиць, штраф за серію, виведення з обігу за невдачами (поріг 3 у холодній фазі, 5 при сильному застої), межі ваг, дві політики звуження кроку.

Еталон Hanoi після кожного з двох кроків окремо: **6.0812 M$ / 1 001 296 симуляцій**, точний збіг.
