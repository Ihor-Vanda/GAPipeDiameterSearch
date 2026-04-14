# Water Distribution Network (WDN) Pipe Diameter Optimizer — Документація

> **Мова коду:** Python 3.10+  
> **Парадигма:** Ітеративний локальний пошук (ILS) з Island-Model паралелізмом  
> **Призначення:** Мінімізація капітальної вартості мережі водопостачання при дотриманні обмежень на мінімальний тиск

---

## Зміст

1. [Архітектура проекту](#1-архітектура-проекту)
2. [Як працює алгоритм — покроково](#2-як-працює-алгоритм--покроково)
3. [Опис модулів](#3-опис-модулів)
4. [Налаштування параметрів](#4-налаштування-параметрів)
5. [Виявлені баги та логічні помилки](#5-виявлені-баги-та-логічні-помилки)
6. [Невикористовувані змінні та мертвий код](#6-невикористовувані-змінні-та-мертвий-код)
7. [Запуск проекту](#7-запуск-проекту)

---

## 1. Архітектура проекту

```
├── main.py                                 — CLI-точка входу, orchestrates multiprocessing pool
├── gui.py                                  — Tkinter/CustomTkinter GUI (альтернативна точка входу)
├── water_sim.py                            — EPANET C-API обгортка (симулятор гідравліки)
├── analytical_solver/orchestrator.py       — SeedFactory, IslandWorker, AnalyticalSolver
├── analytical_solver/context.py            — SolverContext (кеш симуляцій, граф, Dijkstra)
├── analytical_solver/local_search.py       — LocalSearch (gradient_squeeze, heal_network, swap_search)
├── analytical_solver/kicks.py              — KickStrategies (9 стратегій диверсифікації)
├── analytical_solver/pool.py               — SolutionPool (tabu-пам'ять, Hamming-відстані)
├── analytical_solver/cache.py              — LRUCache (OrderedDict-реалізація)
├── analytical_solver/fast_math.py          — Numba JIT-функції (Dijkstra, Hamming, crossover)
└── plot.py                                 — Генерація звітів, CSV, карт топології, графіків збіжності
```

### Залежності між модулями

```
main.py / gui.py
    └── AnalyticalSolver (orchestrator.py)
            ├── SolverContext (context.py)
            │       ├── LRUCache (cache.py)
            │       └── fast_dijkstra (fast_math.py)
            ├── LocalSearch (local_search.py)
            ├── KickStrategies (kicks.py)
            ├── SolutionPool (pool.py)
            │       └── fast_hamming_distance (fast_math.py)
            └── SeedFactory (orchestrator.py)
```

---

## 2. Як працює алгоритм — покроково

### Фаза 0 — Ініціалізація системи

1. **Завантаження конфігурації** (`main.py → load_config`):
   - Читається `*.inp` файл EPANET з топологією мережі.
   - З `costs.csv` завантажуються доступні діаметри (мм або дюйми) та їх вартість за метр.
   - Обидва набори конвертуються в метри СІ та зберігаються у `GAConfig`.

2. **Pre-flight перевірка**: Один тестовий запуск симулятора з нульовими діаметрами для перевірки валідності `.inp`-файлу.

3. **Multiprocessing Pool**: Запускається `N` робочих процесів (за замовчуванням — усі ядра CPU). Кожен процес ізольовано ініціалізує власну копію `WaterSimulator` у тимчасовій директорії.

4. **Калібрування апаратного забезпечення** (`SolverContext._calibrate_simulator`):
   - Виконується 10 пробних симуляцій для вимірювання швидкості (`sim_speed` в сим/сек).
   - Результат використовується для адаптивного тайм-менеджменту.

---

### Фаза 1 — Генерація початкових рішень (SeedFactory)

Для кожного `IslandWorker` на **Епосі 0** генеруються стартові рішення методом `make_diverse_seeds()`:

**Крок 1.1 — Гідравлічний посів (Velocity-Based Seeding)**:

```
Для v ∈ {0.8, 1.0, 1.2} м/с:
    1. Початкове рішення: усі труби на максимальний діаметр
    2. Ітераційна петля (≤10 кроків):
        a. Запуск гідравлічного симулятора → отримати потоки Q[i]
        b. Для кожної труби: d_ideal = √(4Q / πv)
        c. Округлення до найближчого доступного діаметру
        d. Якщо рішення не змінилось → зупинити
    3. Якщо рішення порушує обмеження тиску → heal_network()
    4. gradient_squeeze() для пошуку локального оптимуму
```

**Крок 1.2 — Резервний пул**: 4 випадкові рішення, зцілені та грубо оптимізовані.

На **Епосі N>0** — `make_warm_seeds()` розподіляє ролі між воркерами (система каст):

| Роль       | Частота         | Стратегія                                                         |
| ---------- | --------------- | ----------------------------------------------------------------- |
| EXPLOITER  | ~25% воркерів   | Мікромутації ±1 на кращому архівному рішенні                      |
| RELINKER   | ~25%            | Greedy Path Relinking між двома архівними рішеннями               |
| ARCHITECT  | ~25%            | Консенсусне рішення (значення, однакові у топ-3 архіву) + мутації |
| EXPLORER   | ~25%            | Макромутації ±2                                                   |
| ADVENTURER | останній воркер | Примусово `make_diverse_seeds()` для глобального різноманіття     |

---

### Фаза 2 — Ініціалізація IslandWorker

Кожен воркер запускає метод `run()`:

1. Всі стартові рішення передаються в `_initialize_seeds()`:
   - Для кожного насіння обчислюється `score = cost - (p_surplus × dyn_bonus)`.
   - `dyn_bonus` — динамічний бонус за надлишковий тиск (не дає відкидати рішення з запасом тиску).
   - Валідні рішення ініціалізують `run_best_cost` / `run_best_sol`.

2. Встановлюються параметри поточного епоху (`ipc_immunity = 30`, `stag_limit = 4`).

---

### Фаза 3 — Головний цикл оптимізації (IslandWorker.run)

```
while (elapsed < time_budget AND sim_count < max_sims):
    1. IPC-обмін — читання стану інших воркерів з shared_memory
    2. Rescue-перевірка — якщо воркер відстає >5% від глобального кращого
    3. Swap Search (кожні 8 раундів) — мікрооптимізація поточного рішення
    4. Apply Kick — вибір та виконання стратегії диверсифікації
    5. Generate Mutations — gradient_squeeze() для кандидатів з пулу
    6. Beam Search Update — відбір кращих кандидатів у активний пул
    7. Migration (кожні ~15k симуляцій) — гібридизація з глобально кращим
    8. Stats Decay (кожні 10 раундів) — затухання ваг стратегій × 0.80
```

---

### Фаза 3.1 — Локальний пошук (LocalSearch.gradient_squeeze)

Основний "полірувальник" рішень. Жадібний покроковий спуск:

```
Ініціалізація:
    dyn_bonus = cost_start × 0.001
    score = cost - (p_min - h_min) × dyn_bonus   ← penalized objective

WHILE improved:
    Перемішати порядок труб
    FOR кожна труба idx (не locked):
        Спробувати зменшити діаметр на 1 крок (якщо idx > 0)
        Якщо режим не quick_mode: спробувати збільшити на 1 крок
        Якщо нове рішення feasible І score покращився → прийняти

    Кожні 3 проходи: перевірити мінімальне відносне покращення (> 0.03%)
    Якщо ні → зупинити достроково
```

**Параметри прискорення**:

- `quick_mode=True`: пропускати труби з unit_loss < 0.1 м/м (нездатні до покращення)
- `max_passes=N`: обмежити кількість проходів
- `locked_pipes`: набір заморожених труб (не змінюються)

---

### Фаза 3.2 — Стратегії диверсифікації (Kick Strategies)

При стагнації вибирається одна зі стратегій через **UCB1-подібне зважене випадкове вибирання** (`win_rate = wins/tries`):

| Стратегія                                   | Алгоритм                                                                                            |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **SHOCK** (forcing_hand_kick)               | Збільшити діаметри K найгірших труб на критичному шляху                                             |
| **BOTTLENECK** (upstream_bottleneck_kick)   | Знайти першу «звужуючу» трубу на критичному шляху (де діаметр менший за попередній) та збільшити її |
| **TOPO_INV** (topological_inversion_kick)   | Знайти альтернативний маршрут до критичного вузла і підсилити його                                  |
| **LOOP_BALANCE** (loop_balancing_kick)      | Знизити діаметр одного з кільцевих трубопроводів, переспрямувавши потік                             |
| **ZERO_SUM** (zero_sum_shift_kick)          | Upgrade одна труба / Downgrade декілька — бюджетно нейтральний обмін                                |
| **TRIM** (peripheral_trim_kick)             | Зменшити периферійні труби з низькими втратами                                                      |
| **SMART_PERTURB** (smart_perturbation_kick) | Випадкові зсуви ±1/±2 на N% труб                                                                    |
| **RUIN_RECREATE** (ruin_and_recreate_kick)  | Знищити кластер труб навколо критичного вузла, зцілити                                              |
| **BASIN_ESCAPE** (basin_escape)             | Трансплантувати гени з найбільш несхожого архівного рішення                                         |
| **SPATIAL_PERTURB**                         | Мутувати лише труби в заданому географічному радіусі від випадкового епіцентру                      |

**Параметр T** (Temperature, 0..1) — пропорційний до `stagnation_counter / stag_limit`. Визначає агресивність кіку.

---

### Фаза 3.3 — Відновлення feasibility (LocalSearch.heal_network)

Після будь-якого кіку, що може порушити обмеження тиску:

```
WHILE мережа infeasible:
    1. Знайти критичний вузол (crit_node) з мінімальним тиском
    2. Обчислити домінантний шлях від джерела до crit_node (fast_dijkstra)
    3. Для кожної труби на шляху (яка може бути збільшена):
        - Обчислити ефективність = unit_loss / sqrt(delta_cost)
    4. Збільшити трубу з найвищою ефективністю на 1 крок
    5. Додати її до locked_pipes (не зменшувати потім)
ЯКЩО за N кроків не вдалось → повернути failure
```

---

### Фаза 3.4 — Beam Search та оновлення пулу

`_generate_mutations()` — для кожного рішення в `active_pool`:

1. Запустити `gradient_squeeze()` з `max_passes=2, quick_mode=True`
2. Зібрати кандидатів

`_beam_search_and_update()`:

1. Відсортувати кандидатів за `score`.
2. Прийняти нове рішення, якщо воно краще за поточний `run_best`.
3. Перевірити через `is_ghost_solution()` якщо покращення > 2% (захист від кешових артефактів).
4. Оновити `active_pool`, видаляючи tabu-рішення.

---

### Фаза 4 — IPC та Migration між воркерами

Спільна пам'ять `shared_progress` (multiprocessing.Manager dict):

- `shared_progress[worker_id]` — поточний стан воркера `{round, sims, best_cost}`
- `shared_progress['global_best']` — кращий кандидат серед усіх воркерів
- `shared_progress[f'best_sol_{i}']` — рішення воркера `i` для пасивної ін'єкції
- `shared_progress['global_archive']` — топ-6 різноманітних рішень

**Rescue-механізм**: якщо воркер відстає від глобального кращого на >5%, він скидає свій пул та приймає глобальне рішення.

**Migration**: кожні `max_sims / 20` симуляцій воркер порівнює своє рішення з глобальним:

- T < 0.3: повне копіювання глобального рішення
- 0.3 ≤ T < 0.7: `_spatial_crossover()` — гібрид власного та глобального
- T ≥ 0.7: ігнорується (воркер у стані глибокої розвідки)

---

### Фаза 5 — Фінальна полірування та звітність

1. **Final Polish** (`gradient_squeeze` без обмежень проходів) — застосовується до глобально кращого рішення.
2. **Export** (`plot.py`):
   - `solution_*.csv` — таблиця труб з діаметрами та вартістю
   - `optimized_network.inp` — EPANET файл із оптимальними діаметрами
   - `engineering_report.txt` — детальний звіт по тиску/швидкості/втратах
   - `convergence.png` + `convergence_history.csv`
   - `network_map.png` — кольорова карта топології

---

## 3. Опис модулів

### `water_sim.py` — WaterSimulator

Обгортка навколо EPANET C-API (`wntr.epanet.toolkit.ENepanet`). Всі симуляції виконуються **в оперативній пам'яті** (не пишуть файли на диск).

| Метод                              | Призначення                                                       |
| ---------------------------------- | ----------------------------------------------------------------- |
| `evaluate(individual, pf, eps)`    | Повна оцінка: вартість + штраф за порушення тиску                 |
| `get_stats(individual)`            | Повертає `(cost, min_pressure, max_pressure, critical_node)`      |
| `get_heuristics(individual)`       | Питомі втрати тиску `hl/L` для кожної труби (гіперплощина витрат) |
| `get_hydraulic_state(diameters_m)` | Потоки та feasibility (приймає **метри**, не індекси!)            |

**Важлива деталь**: автоматична адаптація одиниць:

- US Customary (`flow_units < 5`): діаметри → дюйми, тиск × 0.703 PSI→м, втрати × 0.3048 ft→м
- Metric: діаметри → мм, тиск та втрати в метрах без змін

### `context.py` — SolverContext

Центральна структура даних воркера. Містить кеші, граф, Dijkstra-масиви.

| Атрибут                      | Тип             | Призначення                                               |
| ---------------------------- | --------------- | --------------------------------------------------------- |
| `sim_cache`                  | LRUCache(50000) | Кешує результати `get_stats` по ключу `tuple(indices)`    |
| `heuristic_cache`            | LRUCache(50000) | Кешує `get_heuristics`                                    |
| `csr_indptr/indices/weights` | np.array        | CSR-матриця графу для fast_dijkstra                       |
| `source_ids`                 | np.array        | Індекси вузлів-джерел (резервуарів)                       |
| `baseline_bonus`             | float           | Стартовий dyn_bonus на основі середньої різниці вартостей |

### `cache.py` — LRUCache

Стандартна LRU-реалізація на `OrderedDict`. При переповненні (`len > maxsize`) видаляється найстаріший елемент.

### `fast_math.py` — Numba JIT Functions

| Функція                 | Призначення                                    | Складність |
| ----------------------- | ---------------------------------------------- | ---------- |
| `fast_hamming_distance` | Відстань між двома рішеннями                   | O(n)       |
| `fast_avg_hamming`      | Середня Hamming-відстань у пулі                | O(n²)      |
| `fast_dijkstra`         | Дейкстра від множини джерел до цільового вузла | O(V²)      |
| `fast_crossover`        | Рівномірний crossover з імовірністю p_mine     | O(n)       |

### `pool.py` — SolutionPool

Управляє tabu-пам'яттю:

- `tabu_fingerprints`: словник `{fingerprint: round_added}`, tenure=80 раундів
- `basin_tabu`: "відбиток басейну" (огрублена сигнатура) — запобігає повторному дослідженню вже відвіданих зон пошукового простору
- `kick_tabu_set`: заборонені стратегії для поточного стану

### `local_search.py` — LocalSearch

| Метод                   | Коли викликається                             |
| ----------------------- | --------------------------------------------- |
| `gradient_squeeze`      | Після кожного кіку та в beam search           |
| `heal_network`          | Після будь-якого кіку, що може порушити тиск  |
| `swap_search`           | Кожні 8 раундів головного циклу               |
| `get_high_impact_pipes` | Допоміжний — для визначення пріоритетних труб |
| `evaluate_candidate`    | Паралельна оцінка кандидатів (beam)           |

### `kicks.py` — KickStrategies

Всі методи повертають `(kicked_sol, locked_pipes_set, description_string)`.
Деякі методи додатково повертають `failed_pipe_id` (для BOTTLENECK та LOOP_BALANCE).

### `orchestrator.py` — SeedFactory + IslandWorker + AnalyticalSolver

`AnalyticalSolver.solve_standalone()` — головний метод, що:

1. Розбиває час на `N_epochs × time_per_epoch`.
2. Запускає `N_workers` через `multiprocessing.Pool.apply_async()`.
3. Збирає результати, формує глобальний архів.
4. Повторює для наступного епоху з warm seeds.

---

## 4. Налаштування параметрів

### CLI-параметри (`main.py`)

| Аргумент     | За замовчуванням            | Опис                                           |
| ------------ | --------------------------- | ---------------------------------------------- |
| `--inp`      | `InputData/Hanoi/Hanoi.inp` | Шлях до EPANET `.inp` файлу                    |
| `--costs`    | `InputData/Hanoi/costs.csv` | Шлях до таблиці вартостей                      |
| `--hmin`     | `30.0`                      | Мінімально допустимий тиск (м)                 |
| `--units`    | `mm`                        | Одиниці діаметрів: `mm` або `in`               |
| `--cores`    | `0` (всі)                   | Кількість CPU ядер                             |
| `--runs`     | `1`                         | Кількість незалежних запусків                  |
| `--run_mode` | `analytical`                | `analytical` або `fast_analytical`             |
| `--v_opt`    | `1.0`                       | Цільова швидкість потоку (м/с) для посіву      |
| `--max_sims` | `None` (∞)                  | Глобальний бюджет симуляцій                    |
| `--config`   | `None`                      | JSON-файл конфігурації (перезаписує аргументи) |

### Приклад конфігураційного JSON-файлу

```json
{
  "inp": "InputData/MyNetwork/network.inp",
  "costs": "InputData/MyNetwork/costs.csv",
  "hmin": 25.0,
  "units": "mm",
  "cores": 8,
  "runs": 3,
  "v_opt": 1.2,
  "max_sims": 5000000
}
```

### Ключові внутрішні параметри (в `orchestrator.py`)

| Параметр                   | Де задається            | Значення                           | Вплив                             |
| -------------------------- | ----------------------- | ---------------------------------- | --------------------------------- |
| `BEAM_WIDTH`               | `IslandWorker.__init__` | 5 (SMALL/MEDIUM), 8 (LARGE/XLARGE) | Ширина beam search                |
| `stag_limit`               | `IslandWorker.run`      | `4 + 4×progress_ratio`             | Поріг стагнації перед кіком       |
| `ipc_immunity`             | `IslandWorker.run`      | 30 раундів                         | Захист від надто ранньої міграції |
| `_migration_interval_sims` | `IslandWorker.run`      | `max_sims / 20` або 15000          | Частота міжостровної міграції     |
| `dyn_bonus`                | dynamic                 | `best_cost × 0.001 × U(0.7, 1.3)`  | Баланс між вартістю і тиском      |
| `min_rel_improvement`      | `gradient_squeeze`      | `0.0003` (0.03%)                   | Порог зупинки LS                  |

### Параметри кешу

| Кеш               | Розмір         | Де                       |
| ----------------- | -------------- | ------------------------ |
| `sim_cache`       | 50 000 записів | `SolverContext.__init__` |
| `heuristic_cache` | 50 000 записів | `SolverContext.__init__` |

**Для великих мереж (>500 труб)** рекомендується збільшити до 100 000–200 000 рядком:

```python
self.sim_cache = LRUCache(200000)
```

### Налаштування network_class

Клас мережі визначає агресивність пошуку автоматично:

| Клас   | Умова          | Beam Width | Ефект                                         |
| ------ | -------------- | ---------- | --------------------------------------------- |
| SMALL  | n < 50         | 5          | Повний exhaustive search                      |
| MEDIUM | 50 ≤ n < 200   | 5          | Стандартний режим                             |
| LARGE  | 200 ≤ n < 1000 | 8          | Ширший beam, менше комбінацій swap            |
| XLARGE | n ≥ 1000       | 8          | Мінімальні кандидати, тільки швидкі евристики |

---

## 5. Виявлені баги та логічні помилки

### 🔴 BUG-01: `topological_inversion_kick` — некоректний `force_idx`

**Файл**: `kicks.py`, рядок ~123  
**Код**:

```python
force_idx = min(max(target_capacity_idx, self.ctx.max_d_idx - 1), self.ctx.max_d_idx)
```

**Проблема**: `max(target_capacity_idx, self.ctx.max_d_idx - 1)` завжди дорівнює `max_d_idx - 1` або `max_d_idx`, оскільки нижня межа — `max_d_idx - 1`. `target_capacity_idx` — це середній індекс поточного критичного шляху, який зазвичай значно менший. Логіка «підняти альтернативний шлях до рівня потужності домінантного» повністю нівелюється.

**Виправлення**:

```python
force_idx = min(target_capacity_idx + 1, self.ctx.max_d_idx)
```

---

### 🔴 BUG-02: `make_warm_seeds` (RELINKER) — можливий `TypeError: 'NoneType' is not subscriptable`

**Файл**: `orchestrator.py`, рядок ~160  
**Код**:

```python
if len(archive) >= 2:
    best_dist = -1
    target_sol = None          # ← ініціалізація None
    for _, arch_sol in archive:
        dist = sum(...)
        if dist > best_dist and dist > 0:
            best_dist = dist
            target_sol = arch_sol

    diff_indices = [i for i in range(self.ctx.num_pipes) if base_sol[i] != target_sol[i]]  # ← crash!
```

**Проблема**: якщо всі рішення в архіві ідентичні (`dist == 0` для всіх), `target_sol` залишається `None`, і наступний рядок впаде з `TypeError`.

**Виправлення**:

```python
if target_sol is None:
    for _ in range(self.BEAM_WIDTH - 1):
        seeds.append(list(base_sol))
else:
    diff_indices = [...]
    ...
```

---

### 🟡 BUG-03: `WaterSimulator.__del__` — некоректне знищення при помилці `__init__`

**Файл**: `water_sim.py`  
**Проблема**: якщо `__init__` кидає виняток до присвоєння `self.api`, виклик `__del__` призведе до `AttributeError`.

**Виправлення**:

```python
def __del__(self):
    try:
        if hasattr(self, 'api'):
            self.api.ENclose()
    except:
        pass
```

---

### 🟡 BUG-04: `is_ghost_solution` не рахує `sim_count`

**Файл**: `context.py`  
**Проблема**: метод викликає `self.simulator.get_stats(indices)` напряму (не через `get_cached_stats`), тому симуляція не враховується в `self.sim_count`. Це спотворює статистику та умову зупинки `epoch_sims >= self.max_sims`.

**Виправлення**: замінити виклик на `self.get_cached_stats(indices)` (кеш захистить від подвійного рахунку).

---

### 🟡 BUG-05: `WorkerLogger` не реалізує `writelines`

**Файл**: `main.py` та `gui.py`, всередині `analytical_worker_task`  
**Проблема**: клас `WorkerLogger` підміняє `sys.stdout`, але не реалізує метод `writelines`. Якщо будь-яка бібліотека (наприклад, `traceback.print_exc()`) використовує `sys.stdout.writelines()`, виникне `AttributeError`.

**Виправлення**:

```python
def writelines(self, lines):
    for line in lines:
        self.write(line)
```

---

### 🟡 BUG-06: Дублювання коду між `main.py` та `gui.py`

**Проблема**: функції `worker_init`, `worker_eval_task`, `analytical_worker_task` повністю продубльовані. При виправленні помилки в одній копії інша залишається незмінною.

**Рекомендація**: винести в окремий модуль `worker_tasks.py` та імпортувати звідти.

---

## 6. Невикористовувані змінні та мертвий код

### `fast_math.py` — `locked_count` у `fast_crossover`

```python
# Рядок ~47:
locked_count += 1    # ← обраховується, але ніколи не повертається
```

Функція повертає тільки `child`. `locked_count` — мертва змінна.

---

### `orchestrator.py` — `live_best_sol` частково надлишковий

Змінна `live_best_sol` ініціалізується, оновлюється, але у callback `ui_callback` передається `list(global_best_sol)`, а не `live_best_sol`. Логіка синхронізована, але назви вводять в оману.

---

### `pool.py` — Залишений коментар-завдання

```python
from .fast_math import fast_hamming_distance # 🔴 Додайте імпорт
```

Рядок 1: коментар `# 🔴 Додайте імпорт` залишився після того, як імпорт вже було додано. Потрібно видалити.

---

### `kicks.py` — Дублювання імпортів у `spatial_perturb_kick`

```python
def spatial_perturb_kick(self, indices, T, **kwargs):
    import random          # ← вже імпортовано на рівні модуля
    import networkx as nx  # ← вже імпортовано на рівні модуля
```

Ці рядки — зайві. Видалити.

---

### `kicks.py` — `feas` не використовується у `zero_sum_shift_kick`

```python
_, _, feas, crit_node = self.ctx.get_cached_stats(indices)
```

Змінна `feas` отримується, але ніде не перевіряється в тілі методу.

---

### `orchestrator.py` — `WaterSimulator.worker_eval_wrapper` (застарілий код)

У `main.py`:

```python
WaterSimulator.worker_eval_wrapper = staticmethod(worker_eval_task)
```

Атрибут `worker_eval_wrapper` ніде не викликається у показаному коді. Ймовірно, залишок від попередньої архітектури.

---

## 7. Запуск проекту

### CLI-запуск

```bash
# Базовий запуск на тестовій мережі Hanoi
python main.py --inp InputData/Hanoi/Hanoi.inp --costs InputData/Hanoi/costs.csv --hmin 30

# З обмеженням бюджету симуляцій та конкретною кількістю ядер
python main.py --inp network.inp --costs costs.csv --hmin 25 --cores 4 --max_sims 2000000

# Швидкий режим (менше проходів локального пошуку)
python main.py --inp network.inp --costs costs.csv --run_mode fast_analytical

# З JSON-конфігом
python main.py --config my_config.json
```

### GUI-запуск

```bash
python gui.py
```

### Формат `costs.csv`

```csv
diameter,cost_per_meter
100,45.72
150,70.40
200,98.39
250,129.33
...
```

Перший стовпець — діаметр у мм (або дюймах, якщо `--units in`).  
Другий стовпець — вартість прокладання на метр довжини (будь-яка валюта, але однакова для всіх).

### Виведення результатів

```
OutputDataExperiments/
└── 2025-01-15_10-30-00/
    ├── logs/
    │   ├── run_2025-01-15_10-30-00.txt     ← головний лог
    │   ├── worker_01.txt                    ← лог воркера 1
    │   └── worker_02.txt
    ├── plots/
    │   ├── convergence.png
    │   └── network_map.png
    └── tables/
        ├── solution_champion.csv            ← оптимальна таблиця труб
        ├── optimized_network.inp            ← EPANET-файл
        ├── engineering_report.txt           ← детальний інженерний звіт
        ├── convergence_history.csv
        └── runs_summary.csv
```
