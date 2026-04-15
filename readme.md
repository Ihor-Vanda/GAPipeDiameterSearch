# Water Distribution Network (WDN) — Алгоритм оптимізації діаметрів

> **Мова коду:** Python 3.10+ (Numba JIT, EPANET C-API, multiprocessing)  
> **Парадигма:** Iterated Local Search (ILS) + Island Model паралелізм + UCB1-адаптація стратегій  
> **Задача:** мінімізація капітальної вартості мережі водопостачання при виконанні обмеження на мінімальний тиск у всіх вузлах

---

## Зміст

1. [Архітектура проекту](#1-архітектура-проекту)
2. [Модель задачі — формальна постановка](#2-модель-задачі--формальна-постановка)
3. [Шар симуляції — EPANET C-API](#3-шар-симуляції--epanet-c-api)
4. [SolverContext — центральна структура даних](#4-solvercontext--центральна-структура-даних)
5. [Генерація початкових рішень (SeedFactory)](#5-генерація-початкових-рішень-seedfactory)
6. [Локальний пошук (LocalSearch)](#6-локальний-пошук-localsearch)
7. [Головний цикл оптимізації (IslandWorker.run)](#7-головний-цикл-оптимізації-islandworkerrun)
8. [Механізм диверсифікації — Kick Strategies](#8-механізм-диверсифікації--kick-strategies)
9. [Самоадаптивні параметри кіків (μ-Learning)](#9-самоадаптивні-параметри-кіків-μ-learning)
10. [Паралелізм та Island Model](#10-паралелізм-та-island-model)
11. [Фінальна полірування та звітність](#11-фінальна-полірування-та-звітність)
12. [Налаштування параметрів](#12-налаштування-параметрів)

---

## 1. Архітектура проекту

```
├── main.py                                  — CLI-точка входу, multiprocessing pool
├── gui.py                                   — Tkinter GUI (альтернативна точка входу)
├── water_sim.py                             — EPANET C-API обгортка (WaterSimulator)
├── analytical_solver/orchestrator.py        — SeedFactory · IslandWorker · AnalyticalSolver
├── analytical_solver/context.py             — SolverContext (кеш, граф, CSR-Dijkstra)
├── analytical_solver/local_search.py        — LocalSearch (gradient_squeeze · heal · swap)
├── analytical_solver/kicks.py               — KickStrategies (10 стратегій диверсифікації)
├── analytical_solver/pool.py                — SolutionPool (tabu · basin_tabu · Hamming)
├── analytical_solver/cache.py               — LRUCache (OrderedDict, O(1) get/set)
├── analytical_solver/fast_math.py           — Numba JIT (Dijkstra · Hamming · crossover)
└── plot.py                                  — CSV · INP · звіт · convergence.png · map.png
```

### Граф залежностей

```
main.py / gui.py
    └── AnalyticalSolver
            ├── SolverContext ──► LRUCache, fast_dijkstra (Numba)
            ├── SeedFactory   ──► LocalSearch, SolverContext
            ├── IslandWorker  ──► SolutionPool, KickStrategies, LocalSearch
            │       └── SolutionPool ──► fast_hamming_distance (Numba)
            └── WaterSimulator (EPANET C-API, wntr)
```

---

## 2. Модель задачі — формальна постановка

### Змінні рішення

Мережа водопостачання містить `n` трубопроводів. Кожна труба `i` отримує діаметр з дискретного каталогу розмірів:

```
d_i ∈ {D_0, D_1, ..., D_K}   D_j < D_{j+1}, одиниці — метри СІ
```

Алгоритм оперує **цілочисельними індексами** `x_i ∈ {0, 1, ..., K}`, де `K = max_d_idx`.  
Рішення = вектор `x = [x_0, x_1, ..., x_{n-1}]`.

### Цільова функція (вартість)

```
C(x) = Σ_i  L_i · cost(D_{x_i})
```

де `L_i` — довжина труби `i` (метри), `cost(D)` — вартість за метр для діаметру `D` ($/м) із `costs.csv`.

### Обмеження

```
p_j(x) ≥ h_min   для всіх вузлів-споживачів j
```

де `p_j` — тиск у вузлі `j` в метрах водяного стовпа, `h_min` — задається параметром `--hmin`.

### Штрафна функція (лише для `evaluate()`)

```
f(x) = C(x) + penalty_factor · Σ_j max(0, h_min - p_j(x))
```

Використовується тільки в `WaterSimulator.evaluate()` для генетичного алгоритму (legacy). Основний алгоритм оперує **feasibility-aware score**:

```
score(x) = C(x) - p_surplus(x) · dyn_bonus
```

де `p_surplus = min_j(p_j) - h_min` — мінімальний надлишок тиску,  
`dyn_bonus = best_cost × 0.001 × U(0.95, 1.05)` — стохастичний бонус за тиск.

Рішення прийнятне якщо `p_surplus ≥ 0` (або `≥ -0.5` в epsilon-relaxed режимі beam search).

---

## 3. Шар симуляції — EPANET C-API

### Чому C-API, а не Python WNTR

`wntr.sim.EpanetSimulator` щоразу записує та перечитує файли. C-API (`ENepanet`) тримає мережу у пам'яті між симуляціями, що дає **10–50× прискорення** для коротких серій.

### Ініціалізація (`WaterSimulator.__init__`)

1. `wntr.network.WaterNetworkModel(inp_file)` — читання топології для отримання довжин та назв компонентів (wntr завжди конвертує довжини в метри).
2. `ENepanet(inp_file, rpt_file, bin_file)` → `ENopen()` — відкриття мережі в C-пам'яті.
3. **Автодетекція одиниць**: `flow_units = ENgetflowunits()`
   - `flow_units < 5` → US Customary: `diam × 39.3701` (м→дюйми), тиск `× 0.7032` (PSI→м), втрати `× 0.3048` (ft→м)
   - `flow_units ≥ 5` → Metric: `diam × 1000` (м→мм), тиск і втрати без змін

### Цикл гідравлічного розрахунку (in-memory)

```python
ENopen H()        # відкрити гідравлічний розрахунок
ENinitH(0)        # 0 = не писати результати у файл
while True:
    ENrunH()      # розрахунок одного часового кроку
    tstep = ENnextH()
    if tstep <= 0: break
ENcloseH()
```

Вузлові тиски і швидкості залишаються в C-пам'яті після `ENcloseH()`.

### Методи симулятора

| Метод                          | Що робить              | Повертає                          |
| ------------------------------ | ---------------------- | --------------------------------- |
| `evaluate(x, pf, eps)`         | Повна оцінка з штрафом | `cost + penalty`                  |
| `get_stats(x)`                 | Вартість + гідравліка  | `(cost, p_min, p_max, crit_node)` |
| `get_heuristics(x)`            | Питомі втрати тиску    | `[hl_i/L_i, ...]` м/м             |
| `get_hydraulic_state(diams_m)` | Потоки + feasibility   | `(flows, is_feasible)`            |

`get_stats` — головний метод. Повертає:

- **cost** = Σ L_i × cost[x_i]
- **p_min** = мінімальний тиск серед вузлів-споживачів
- **crit_node** = ім'я вузла з мінімальним тиском (критичний вузол)

---

## 4. SolverContext — центральна структура даних

`SolverContext` ізолює весь стан одного воркера.

### LRU-кеш симуляцій

```python
sim_cache    = LRUCache(maxsize=50_000)   # ключ: tuple(x), значення: (cost, p_min, p_max, crit_node)
heuristic_cache = LRUCache(maxsize=50_000)   # ключ: tuple(x), значення: [hl/L, ...]
```

`get_cached_stats(x)`:

1. Шукає `tuple(x)` в `sim_cache`.
2. При cache miss: `sim_count++`, `simulator.get_stats(x)`, зберігає результат.
3. Попадання: O(1) без жодного звернення до EPANET.

### CSR-граф для Dijkstra

При ініціалізації будується CSR (Compressed Sparse Row) представлення графу мережі для `fast_dijkstra` (Numba):

```python
csr_indptr[i]..csr_indptr[i+1]   — діапазон суміжних вузлів для вузла i
csr_indices[k]                   — вузол-сусід
csr_edge_pipe[k]                 — індекс труби на ребрі (u, v)
```

Ваги ребер обчислюються динамічно: `w_e = 100.0 / (x[pipe_e] + 1.0)` — менший діаметр дає більшу вагу → Dijkstra знаходить шлях через **найширші** труби.

### `get_dominant_path(x, crit_node)`

```
Dijkstra(джерела → crit_node, ваги = 100/(x+1))
→ найкоротший (за inverse-diameter) шлях
→ список індексів труб на цьому шляху
```

Це "домінантний шлях постачання" — найбільш вузьке місце між джерелом і критичним вузлом.

### Калібрування (`_calibrate_simulator`)

10 пробних симуляцій з невеликими варіаціями → вимірює `sim_speed` (сим/сек). Використовується для обчислення `time_limit_sec` коли `max_sims` задано.

### `is_ghost_solution`

Захист від "кешових привидів" — артефактів, де кешований результат дає оптимістичну оцінку, а реальна симуляція дає гірший результат. Викликається лише при поліпшенні > 2%:

```python
def is_ghost_solution(self, x, cost):
    # перевірити кеш → якщо p_min < h_min - 0.01 → це привид
```

---

## 5. Генерація початкових рішень (SeedFactory)

### Метод 1: Velocity-Based Seeding (`make_diverse_seeds`)

Ключова ідея: для заданої цільової швидкості `v` обчислити ідеальний діаметр кожної труби з рівняння нерозривності:

```
d_ideal = sqrt(4 · |Q_i| / (π · v))
```

де `Q_i` — потік через трубу `i` (м³/с).

**Алгоритм (ітеративний)**:

```
Для кожної v ∈ {1.0, 1.2, 0.8} м/с:
  x ← [max_d_idx, ..., max_d_idx]   (старт з максимальних діаметрів)

  Повторювати до 10 разів:
    diams_m = [ctx.diameters[x_i] for i]
    flows, _ = simulator.get_hydraulic_state(diams_m)   ← гідравлічний розрахунок

    Для кожної труби i:
      d_ideal = sqrt(4·|flows[i]| / (π·v))
      x_i_new = bisect_left(ctx.diameters, d_ideal)     ← округлення до каталогу

    Якщо x_new == x: break   ← збіжність
    x ← x_new

  Якщо x infeasible → heal_network(x)
  x ← gradient_squeeze(x, max_passes=12, quick_mode=True)
  seeds.append(x)
```

Три різні швидкості дають три різні початкові точки в просторі рішень, що відповідають різним режимам роботи мережі.

### Метод 2: Backbone Seeding (`make_backbone_seed`)

Стратифікований підхід: труби сортуються за потоком `|Q_i|` і поділяються на три категорії:

```
Топ 20% за потоком (магістральні):   x_i = max_d_idx
Решта 30% (транзитні):                x_i = ideal_d(Q_i, v=1.2)
Низ 50% за потоком (периферійні):    x_i = 0
```

Потім `heal_network` відновлює feasibility, `gradient_squeeze` оптимізує.

### Метод 3: Fast Sweep (`make_diverse_seeds_for_fast`)

16 різних швидкостей `v ∈ {0.5, 0.6, ..., 2.0}` з 5 ітераціями кожна. Використовується лише в `solve_fast()` режимі.

### Метод 4: Reserve Pool (`make_reserve_pool`)

4 випадкові рішення `x_i ~ Uniform(0, max_d_idx)`, зцілені та грубо оптимізовані. Використовуються як резерв для `_emergency_pool_diversity`.

### Warm Seeds для наступних епох (`make_warm_seeds`)

На Epoch > 0 кожному воркеру призначається роль **касти** на основі `worker_id % 4`:

| Роль           | Умова                  | Стратегія генерації                                                     |
| -------------- | ---------------------- | ----------------------------------------------------------------------- |
| **EXPLOITER**  | `adjusted_id % 4 == 0` | `x_base` + `n_pipes//10` мікромутацій ±1                                |
| **RELINKER**   | `adjusted_id % 4 == 1` | Greedy Path-Relinking між `x_base` та найвіддаленішим архівним рішенням |
| **ARCHITECT**  | `adjusted_id % 4 == 2` | Консенсус топ-3 архіву + `n_pipes//15` мутацій ±2                       |
| **EXPLORER**   | `adjusted_id % 4 == 3` | `x_base` + `n_pipes//5` макромутацій ±2                                 |
| **ADVENTURER** | останній воркер        | Примусово `make_diverse_seeds()` (глобальна різноманітність)            |

### Greedy Path-Relinking (`greedy_path_relink`)

```
diff = [i : x_A[i] ≠ x_B[i]]
Перемішати diff випадково

current = x_A
best_intermediate = x_A
best_cost = cost(x_A)

Для кожного pipe_idx в diff:
    test = current; test[pipe_idx] = x_B[pipe_idx]

    Якщо test feasible AND cost(test) < best_cost:
        best_cost = cost(test); best_intermediate = test

    Якщо test feasible: current = test

Повернути best_intermediate
```

Ключова відмінність від стандартного path-relinking: зберігається не кінцева точка `x_B`, а найкраща проміжна точка на шляху.

---

## 6. Локальний пошук (LocalSearch)

### `gradient_squeeze` — основний польоровщик

**Ідея**: жадібний покроковий спуск по пенальній цільовій функції `score = cost - p_surplus · dyn_bonus`.

```
Ініціалізація:
  Якщо dyn_bonus не заданий: dyn_bonus = cost_start × 0.001
  score_best = cost - p_surplus × dyn_bonus
  milestone_cost = cost; milestone_pass = 0

WHILE improved AND (not max_passes OR passes ≤ max_passes):
  improved = False
  passes++

  Перемішати порядок активних труб (виключаючи locked_pipes)

  ── Рання зупинка кожні 3 проходи ──
  if passes - milestone_pass ≥ 3:
    rel_improvement = (milestone_cost - cost) / milestone_cost
    if rel_improvement < min_rel_improvement (0.0003 = 0.03%): break
    оновити milestone

  Для кожної труби idx:
    ── quick_mode: пропустити "тихі" труби ──
    if quick_mode AND unit_losses[idx] < 0.1: continue

    ── Спроба downgrade (зменшити діаметр) ──
    if x[idx] > 0:
      test[idx] = x[idx] - 1
      (c, p, feas) = get_cached_stats(test)
      if feas AND p ≥ h_min:
        new_score = c - (p - h_min) × dyn_bonus
        if new_score < score_best: прийняти

    ── Спроба upgrade (збільшити діаметр) — лише без quick_mode ──
    if not quick_mode AND x[idx] < max_d_idx:
      test[idx] = x[idx] + 1
      (c, p, feas) = get_cached_stats(test)
      if feas AND p ≥ h_min:
        new_score = c - (p - h_min) × dyn_bonus
        if new_score < score_best: прийняти

Повернути поточний x
```

**Параметри виклику:**

| Параметр              | Значення                | Ефект                                         |
| --------------------- | ----------------------- | --------------------------------------------- |
| `quick_mode=True`     | пропускати `hl/L < 0.1` | 3–5× швидше, менш ретельно                    |
| `max_passes=N`        | обмеження ітерацій      | N=1 → швидка перевірка, N=None → до збіжності |
| `locked_pipes`        | set індексів            | заморожені труби не змінюються                |
| `min_rel_improvement` | 0.0003                  | рання зупинка при мікрокроках                 |

### `heal_network` — відновлення feasibility

```
WHILE infeasible:
  crit_node = вузол з мінімальним тиском
  path_pipes = get_dominant_path(x, crit_node)   ← Dijkstra

  Для кожної труби idx на path_pipes (не в locked):
    Розрахувати ефективність збільшення:
      Якщо n ≥ 200:
        efficiency = unit_losses[idx] / sqrt(delta_cost · abs_cost)   ← нормований
      Інакше:
        efficiency = unit_losses[idx]

  best_pipe = argmax(efficiency)
  x[best_pipe] += 1
  locked.add(best_pipe)
  boosts++

  Якщо candidates пустий: return x, False, boosts

return x, True, boosts
```

Кожне збільшення фіксується в `locked` — гарантує монотонне зростання діаметрів на критичному шляху без регресу.

### `swap_search` — мікрооптимізація (кожні 8 раундів)

Комбінаторний пошук "обмін вартості":

1. Знайти "ліниві" труби (низькі `unit_losses`) → кандидати на downgrade.
2. Знайти труби на критичному шляху → кандидати на upgrade.
3. Спробувати downgrade ленивих.
4. Для малих мереж (≤200 труб): спробувати `upgrade[i] + downgrade[j] + downgrade[k]` трійки.

### `evaluate_candidate` — атомарна оцінка кандидата

```python
test_sol = base + upgrade/downgrade pipe(s)
squeezed = gradient_squeeze(test_sol, locked=upgraded_pipes, max_passes=3, quick_mode=True)
(cost, p_min, _, _) = get_cached_stats(squeezed)

if not feasible OR p_min < h_min: return inf, -inf, None
score = cost - (p_surplus × dyn_bonus)
return score, cost, squeezed
```

Використовується в `_generate_mutations` для паралельної оцінки сусідніх рішень.

---

## 7. Головний цикл оптимізації (IslandWorker.run)

### Ініціалізація стану воркера

```python
self.run_best_cost = float('inf')
self.run_best_sol  = None
self.stagnation_counter = 0
self.stag_limit = 4                 # адаптивно зростає до 12
self.progress_ratio = 0.0           # 0 → 1 протягом часового бюджету
self.is_late_game = False           # True коли progress > 0.5

# Самоадаптивні параметри кіків:
self.mu_ruin_pct    = 0.05   # оптимальний розмір R&R кластеру
self.mu_perturb_pct = 0.10   # SMART_PERTURB
self.mu_spatial_pct = 0.20   # SPATIAL_PERTURB
self.mu_escape_pct  = 0.20   # BASIN_ESCAPE
```

### Адаптивні параметри за часом

```python
progress_ratio = elapsed / time_budget   # або epoch_sims / max_sims
is_late_game   = progress_ratio > 0.5
stag_limit     = 4 + int(8 × progress_ratio)   # 4 → 12
dyn_bonus      = min(run_best, global_best) × 0.001 × U(0.95, 1.05)
```

`stag_limit` зростає з часом — на пізніх стадіях алгоритм "терпіть" більше раундів без прогресу перед запуском агресивних стратегій.

### Температура T (аналог simulated annealing)

```python
T = min(1.0, stagnation_counter / (stag_limit × 3.0))
```

`T ∈ [0, 1]` — показник "відчаю". При `T = 0`: алгоритм у режимі активної оптимізації. При `T = 1`: максимальна стагнація, агресивна диверсифікація.

| T         | Інтерпретація            | Активні стратегії                                       |
| --------- | ------------------------ | ------------------------------------------------------- |
| 0.0 – 0.5 | Активний локальний пошук | SHOCK, BOTTLENECK, LOOP_BALANCE, ZERO_SUM, TRIM         |
| 0.5 – 0.9 | Помірна стагнація        | SPATIAL_PERTURB, SMART_PERTURB, RUIN_RECREATE, TOPO_INV |
| 0.9 – 1.0 | Глибока стагнація        | BASIN_ESCAPE, SPATIAL_PERTURB, RUIN_RECREATE            |

### Структура одного раунду

```
Раунд round_idx:

1.  IPC     → _process_ipc()          читати стан інших воркерів
2.  RESCUE  → _check_rescue()         перевірити відставання від global_best
3.  UPDATE  → shared_progress[id]     публікувати поточний стан
4.  PROGRESS → оновити progress_ratio, is_late_game, stag_limit
5.  RESTART → _check_mini_restart()   якщо дуже довга стагнація
6.  TABU    → kick_tabu_set.clear()   кожні 6 раундів
7.  SWAP    → _apply_swap()           кожні 8 раундів
8.  KICK    → _apply_kick()           якщо stagnation_counter ≥ 2
9.  FLUSH   → _force_flush_next       якщо BASIN_ESCAPE спрацював
10. MUTATE  → _generate_mutations()
11. BEAM    → _beam_search_and_update()
12. MIGRATE → _spatial_crossover()    кожні migration_interval симуляцій
```

### `_generate_mutations` — beam-мутації

Для кожного рішення в `active_pool`:

- Обчислити `unit_losses = get_cached_heuristics(x)` → сортування труб
- `high_friction` (кандидати на upgrade): труби з найвищими `hl/L`
- `low_friction` (кандидати на downgrade): труби з найнижчими `hl/L`
- Адаптивні ліміти за надлишком тиску `p_surplus`:

```python
if p_surplus > 10: downgrade_limit = 15, upgrade_limit = SINGLE_CANDIDATES
elif p_surplus < 2: downgrade_limit = 3,  upgrade_limit = SINGLE_CANDIDATES // 2
else:               downgrade_limit = 8,  upgrade_limit = SINGLE_CANDIDATES
```

`SINGLE_CANDIDATES` залежить від класу мережі:

```python
{"SMALL": n//5, "MEDIUM": n//10, "LARGE": n//20, "XLARGE": 10}
```

Для LARGE/XLARGE — додатково фільтрувати через `get_high_impact_pipes(top_k=n//5)`.

Також генеруються парні апгрейди `(p1, p2)` для труб із `high_friction[:5]`.

### `_beam_search_and_update` — відбір рішень

```
min_dist = max(1, (n × 0.05) × (1 - progress_ratio)³)
  → мінімальна Hamming-відстань між рішеннями у пулі
  → зменшується до 0 наприкінці (дозволяє схожі рішення)

Для кожного кандидата (rank, score, cost, sol):
  1. Пропустити якщо tabu
  2. Уточнення:
       rank == 0: gradient_squeeze(max_passes=5, quick_mode=not(round%2==0))
       rank ∈ 1..2: gradient_squeeze(max_passes=3, quick_mode=True)
       rank ≥ 3: без уточнення
  3. Перевірка різноманітності: hamming(sol, peer) ≥ min_dist для всіх peers
  4. Якщо cost < run_best AND diff > 0.5%: found_new_record = True, stagnation=0
  5. Якщо diff ≤ 0.5% (мікрокрок): stagnation -= 2 (не скидати)

dynamic_beam = max(3, BEAM_WIDTH × (1 + 0.5 × (1 - progress_ratio)))
  → на початку пул ширший, наприкінці звужується

active_pool = unique_next_pool[:dynamic_beam]
```

**Epsilon-relaxation**: рішення з `p_surplus ∈ [-0.5, 0)` потрапляють у пул з пенальтним score `= cost + |p_surplus| × dyn_bonus × 50.0`, але не оголошуються рекордом.

### `_emergency_pool_diversity`

Після кожного beam update:

```python
avg_dist = fast_avg_hamming(pool_matrix)  # середня Hamming між всіма парами
diversity_threshold = (n // 8) × (1 - progress_ratio)

Якщо avg_dist < threshold AND reserve_pool не пустий:
    замінити останнє рішення пулу на резервний seed
```

### `_check_mini_restart` — повний перезапуск воркера

```python
multiplier = 6 (LARGE) або 10 (SMALL/MEDIUM), / 2 якщо is_late_game

Якщо stagnation_counter ≥ stag_limit × multiplier:
    очистити tabu, pool, kick_tabu
    згенерувати 4 мутанти від global_best з n_perturb = min(30..45, n//5) зсувами ±2
    heal → squeeze → додати у пул
    stagnation_counter = 0
```

---

## 8. Механізм диверсифікації — Kick Strategies

### UCB1 вибір стратегії

```python
exploration_C = 0.15 + 0.25 × T   # T=0: 0.15; T=1: 0.40

score(s) = wins[s] / tries[s]  +  C × √(ln(n_total) / tries[s])
         ← exploitation term   ←  exploration term

strategy = argmax score(s) серед valid_strats
```

Ця формула балансує між вибором стратегій, що раніше приносили результат, та дослідженням менш випробуваних.

**consecutive_fails захист**: якщо стратегія провалилась `max_fails` разів поспіль (`3` при T<0.5, `5` при T≥0.5) — виключається з `valid_strats` до першого успіху.

**При провалі кіку**: `strat_wins[s] × 0.90` — швидка деградація рейтингу.

**При успіху**: `reward = 10 × improvement_pct × 100` + decay всіх стратегій × 0.97 (відносна переоцінка переможця).

### `_apply_kick` — повний pipeline

```
1. ВИБІР ЦІЛІ для кіку:
   T ≥ 0.8: kick_target = найбільш несхоже рішення в пулі (max Hamming від run_best)
   T ≥ 0.4: kick_target = random.choice(active_pool)
   T < 0.4: round % 3:
     0 → run_best_sol
     1 → active_pool[0]
     2 → random.choice(active_pool)

2. ВИКОНАННЯ кіку → (forced_sol, locked, log_msg, used_pct)

3. РАННЬА ВІДМОВА:
   raw_deficit > catastrophic_limit = max(20, h_min × (1 + T)):
       відкинути без heal

4. HEAL (якщо потрібно):
   heal_locks = set() для SPATIAL/SMART/RUIN (дозволяємо heal гнучко)
   heal_locks = locked для решти
   heal_network(forced_sol, heal_locks) → якщо fails → discard

5. SQUEEZE:
   BASIN_ESCAPE: без squeeze (пряма ін'єкція)
   Інші:
     quick_passes = max(1, 2 - int(T × 2))    # T=0→2; T=0.5→1; T=1→1
     quick_sol = gradient_squeeze(quick_passes, quick_mode=True)

     HYPERBAND оцінка:
       hb_margin = 0.03 - 0.02 × progress_ratio   # 3% → 1% наприкінці
       is_promising = quick_f AND quick_cost < run_best × (1 + hb_margin)

       Якщо promising:
         gap = (quick_cost - run_best) / run_best
         deep_passes:
           T ≥ 0.6 → 1
           T ≥ 0.4 → 2
           gap < -0.01% → 8 (LARGE) або 5 (SMALL)
           gap ≤ 0.2% → 3
           gap ≤ 2% → 2
           gap ≤ 5% → 2
           інше → 1

         CONSENSUS FREEZE (T < 0.2, is_late_game, archive ≥ 3):
           raw_consensus = {i : всі топ-3 архіву мають однакове x_i}
           freeze_pct = 25% (progress>0.6) або 10%
           max_frozen = n × freeze_pct; 0 якщо progress > 0.88
           consensus_locked = обмежений набір frozen pipes

         final_sol = gradient_squeeze(quick_sol, deep_passes, locked=consensus_locked ∪ locked_for_squeeze)

6. ПРИЙНЯТТЯ рішення:
   deficit = max(0, h_min - p_final)
   is_relaxed_valid = feas AND deficit ≤ 0.5 × T

   effective_cost = cost + deficit × (run_best × 0.10)   # пенальтя
   explosion_threshold = run_best × (1.5 + 0.5 × T)      # відкинути вибухи

   score = effective_cost - p_surplus × dyn_bonus   (якщо feasible)
   score = effective_cost                           (якщо relaxed)

   if effective_cost < run_best:        → Direct Record
   elif strategy == BASIN_ESCAPE:       → Pool Flush
   elif effective_cost < water_level:   → Add to Pool
   elif T ≥ 0.95:                       → HAIL MARY (Accept anyway)
   else:                                → Reject

   water_level = run_best × (1 + 0.03 + 0.07 × T)   # 3%→10% від run_best
```

### Детальний опис кожної стратегії

#### SHOCK (`forcing_hand_kick`) — T < 0.5

```
aggressiveness = 0.05 + 0.35 × T   # 5%→40% труб критичного шляху
limit = max(1, len(path_pipes) × aggressiveness)
Збільшити limit труб з найвищими unit_losses на критичному шляху
```

#### BOTTLENECK (`upstream_bottleneck_kick`) — T < 0.5

```
Фаза 1 — Taper Detection:
  Пройти path_pipes зворотно
  Знайти першу трубу де x[curr] < x[prev] (звуження)
  → збільшити на 1, заблокувати

Фаза 2 — Fallback (якщо звуження нема):
  search_depth = len(path) × (0.3 + 0.4 × T)
  boost_pct = 0.05 + 0.15 × T
  Підняти boost_pct% найгірших труб у search_depth
```

Труби з failed_pipes пропускаються (tabu tenure = 10 раундів).

#### TOPO_INV (`topological_inversion_kick`) — T ∈ [0.5, 0.9)

```
Знайти 5 альтернативних шляхів до crit_node (різні ваги ребер)
Відкинути шляхи з tabu-підписом
Вибрати шлях з мінімальним overlap з домінантним шляхом

aggressiveness = 0.10 + 0.50 × T
max_pipes = len(alt_path) × aggressiveness

Підняти всі вибрані труби до target_capacity_idx = mean(x_i on dominant path)
```

#### LOOP_BALANCE (`loop_balancing_kick`) — T < 0.5

```
cycles = nx.cycle_basis(G)   (кешується)
restrict_pct = 0.10 + 0.30 × T   # частка труб у циклі

Для кожного циклу:
  Вибрати restrict_pct% труб
  Спробувати знизити їх на drop = 1..max_drop
  heal + quick_squeeze → перевірити вартість
  Зібрати кандидатів (до 5)

Повернути random.choice(top-3 кандидатів)
LOOP_BALANCE_PIPE_TENURE = min(80, n // 5)
```

Для деревоподібних мереж автоматично виключається.

#### ZERO_SUM (`zero_sum_shift_kick`) — T < 0.5

```
Для кожної труби i:
  upgrades: c_up = L_i × (cost[x_i+1] - cost[x_i])
            score = unit_loss[i] / c_up   ← bang per buck

  downgrades: c_down = L_i × (cost[x_i] - cost[x_i-1])
              score = c_down / unit_loss[i]   ← ніщо не втрачаємо

search_pool_size = max(15, n // 10)
tests_limit = 5 (T<0.3) або 2 (T≥0.3)

Для кожного апгрейду зі списку:
  Накопичувати downgrades поки savings > cost_invest
  heal → якщо ok та cost < best: зберегти

Труби в zero_sum_tabu (tenure 15 раундів) пропускати
```

#### TRIM (`peripheral_trim_kick`) — T < 0.5

```
periphery = труби поза домінантним шляхом з x_i ≤ 0.6 × max_d_idx
  сортовані за зростанням unit_losses (найтихіші)

trim_pct = 0.02 + 0.13 × T   # 2%→15% периферії
combo = random.sample(periphery[:combo_limit], pipes_to_cut)

25 спроб → heal → evaluate → вибрати random.choice(top-3)
```

#### SMART_PERTURB (`smart_perturbation_kick`) — T ∈ [0.5, 0.9)

```
μ = mu_perturb_pct + 0.15 × T     # самоадаптивне μ
σ = 0.02 + 0.05 × T
target_pct ~ Gauss(μ, σ), clip to [0.01, 0.30]

n_perturb = min(max_perturb, n × target_pct)
max_perturb = {SMALL:∞, MEDIUM:25, LARGE:40, XLARGE:60}

Вибрати n_perturb труб (переважно 0 < x < max_d)
Якщо T > 0.6: delta ~ {-2,-1,+1,+2} (зважено до ±1)
Інакше: delta ~ {-1, +1}

heal → return used_pct (для μ-learning)
```

#### RUIN_AND_RECREATE — T ∈ [0.5, 0.9)

```
Епіцентр:
  T < 0.8: crit_node
  T ≥ 0.8: random node (повне руйнування)

μ_ruin = mu_ruin_pct + 0.10 × T
σ_ruin = 0.01 + 0.04 × T
target_pct ~ Gauss(μ_ruin, σ_ruin), clip to [0.01, 0.25]

cutoff = {n<50: 3, n<200: 4, n<1000: 6, n≥1000: 8}   # адаптивний
Зібрати cluster_pipes ближніх труб через BFS з cutoff

max_drop = min(temp_drop, max_d_idx // 3)   # обмеження каталогом
  temp_drop = 1 + int(2 × T)

Для кожної труби в кластері: x_i -= random.randint(1, max_drop)
heal(kicked, locked=set())   # без обмежень для heal
return used_pct (для μ-learning)
```

#### BASIN_ESCAPE (`basin_escape`) — T ≥ 0.9

```
Знайти найвіддаленіше рішення в global_archive (max Hamming)
Потрібно best_dist ≥ max(2, n × 0.03)

μ_esc = mu_escape_pct + 0.30 × T   # 0.20 + 0.30 → 0.50 при T=1
σ_esc = 0.05 + 0.05 × T
target_pct ~ Gauss(μ_esc, σ_esc), clip to [0.05, 0.60]

n_replace = len(diff_pipes) × target_pct
Замінити n_replace труб значеннями з diverse_sol

heal → pool.basin_tabu.append(run_best_sol signature)
   (поточний басейн позначається як досліджений)

Якщо accepted: active_pool.clear() + forced_flush
               run_best = effective_cost
               ipc_immunity = 50
return used_pct (для μ-learning)
```

#### SPATIAL_PERTURB — T ≥ 0.5

```
base_radius = {n<50: 2, n<200: 3, n<1000: 5, n≥1000: 8}
radius = base_radius + int(base_radius × T)

Вибрати випадковий epicenter
local_nodes = BFS від epicenter з cutoff=radius
local_pipes = всі труби суміжні з local_nodes

μ_sp = mu_spatial_pct + 0.15 × T   # незалежний параметр від SMART!
σ_sp = 0.02 + 0.05 × T
target_pct ~ Gauss(μ_sp, σ_sp), clip to [0.01, 0.40]

target_mutations = min(max_perturb, len(local_pipes) × target_pct)
Мутувати вибрані труби, решта → locked

Fallback на SMART_PERTURB якщо local_pipes порожній
return used_pct (для μ-learning)
```

---

## 9. Самоадаптивні параметри кіків (μ-Learning)

### Концепція

Замість фіксованого `pct = 0.02 + 0.18 × T` кожна стратегія навчається оптимальному розміру втручання через **exponential moving average**:

```
Якщо кік → прямий рекорд (direct record):
    μ_new = 0.9 × μ_old + 0.1 × used_pct
```

де `used_pct` — частка труб, яку фактично зачепив даний кік.

### Параметри та початкові значення

| Параметр         | Стратегія       | μ₀   | Діапазон sampling          | Ефект при зростанні        |
| ---------------- | --------------- | ---- | -------------------------- | -------------------------- |
| `mu_ruin_pct`    | RUIN_RECREATE   | 0.05 | Gauss(μ+0.10T, 0.01+0.04T) | більший кластер руйнування |
| `mu_perturb_pct` | SMART_PERTURB   | 0.10 | Gauss(μ+0.15T, 0.02+0.05T) | більше збурених труб       |
| `mu_spatial_pct` | SPATIAL_PERTURB | 0.20 | Gauss(μ+0.15T, 0.02+0.05T) | більше мутацій у патчі     |
| `mu_escape_pct`  | BASIN_ESCAPE    | 0.20 | Gauss(μ+0.30T, 0.05+0.05T) | більше "генів" від донора  |

**Важливо**: `mu_perturb_pct` і `mu_spatial_pct` — **різні параметри** (виправлено). SMART і SPATIAL мають різну семантику `used_pct` (частка від усіх труб vs частка від local_pipes).

### Умова оновлення μ

Оновлення відбувається лише при **прямому рекорді** (`effective_cost < run_best_cost`):

- Для RUIN/SMART/SPATIAL — у блоці `"if deficit == 0 and effective_cost < run_best"`
- Для BASIN_ESCAPE — окремо, з виправленою умовою (без зайвої перевірки `effective_cost < run_best` в `else` гілці)

### Стохастика навколо μ

Gaussian sampling забезпечує **exploration навколо поточного μ**:

```
target_pct = max(lower, min(upper, random.gauss(dynamic_mu, dynamic_sigma)))
```

При стагнації (T → 1) `dynamic_mu` збільшується лінійно → алгоритм автоматично переходить до агресивніших кіків.

---

## 10. Паралелізм та Island Model

### Архітектура

```
AnalyticalSolver.solve_standalone()
    └── epoch = 1 (один безперервний прогін)
          └── tasks = N_workers × (diameters, time_budget, global_best, archive, seed, ...)
                 └── mp_pool.apply_async(worker_task, task)
                       └── IslandWorker.run(time_budget, global_best, shared_progress)
```

### Shared Memory (`multiprocessing.Manager().dict()`)

| Ключ                                 | Значення                   | Хто пише                 | Хто читає |
| ------------------------------------ | -------------------------- | ------------------------ | --------- |
| `shared_progress[wid]`               | `{round, sims, best_cost}` | воркер wid               | всі       |
| `shared_progress['global_best']`     | `(cost, sol)`              | воркер що знайшов рекорд | всі       |
| `shared_progress[f'best_sol_{wid}']` | `(cost, sol)`              | воркер wid               | всі       |
| `shared_progress['global_archive']`  | список `(cost, sol)`       | оркестратор (кожні 30s)  | воркери   |

### IPC — пасивна ін'єкція (`_process_ipc`)

Умова прийняття рішення від peer воркера:

```python
peer_cost < run_best × 0.995    # на 0.5% краще
peer_cost < last_injected - 1.0  # нове (не вже бачили)
NOT (is_adventurer AND progress < 0.85)  # adventurer ігнорує до пізньої стадії
NOT (stagnation < stag_limit AND NOT massively_better)  # не перебивати активний прогрес
```

### Rescue механізм (`_check_rescue`)

```python
global_lag = (run_best - global_best) / global_best

if (global_lag > 2% AND stagnation ≥ 2×stag_limit) OR global_lag > 5%:
    Прийняти global_best рішення
    pool.clear → вставити (global_best, score - 1e6)  # пріоритет
    stagnation = 0; kick_tabu.clear()
```

### Migration з 3-dimensional crossover (`_spatial_crossover`)

```python
T_migration = stagnation_counter / (stag_limit × 3)

T < 0.3:  # Cold migration
    hybrid = spatial_crossover(run_best_sol, gb_sol, T=0.25)

    # додатковий peer noise
    peer_sol = шукати серед воркерів sol з cost ≠ gb_cost
    if peer_sol: hybrid = spatial_crossover(hybrid, peer_sol, T=0.05)

    heal → вставити в active_pool[0]; stagnation=0; ipc_immunity=50

T ∈ [0.3, 0.7):  # Warm migration
    hybrid = spatial_crossover(run_best_sol, gb_sol, T)
    if hybrid cost < run_best: прийняти як новий run_best

T ≥ 0.7:  # Exploration Shield — ігнорувати global_best
```

`spatial_crossover` — регіональна трансплантація: вибрати epicenter, зібрати `local_nodes` в BFS-радіусі `2 + 6×T`, замінити `local_pipes` від donor_sol.

### Migration interval (адаптивний)

```python
_migration_interval_sims = max(1000, (max_sims // 20) × (1.0 - 0.6 × progress_ratio))
```

На початку — рідкий обмін (кожні 5% бюджету). Наприкінці — частіший (кожні 2%).

### Глобальний архів (`_build_diverse_archive`)

Після закінчення кожного epoch оркестратор будує топ-6 архів:

1. Топ-2 рішення за вартістю (elite).
2. Решта: додати якщо `min_hamming_to_archive ≥ max(15, min(45, n × 0.08))`.

Цей архів передається воркерам наступного epoch для `make_warm_seeds`.

---

## 11. Фінальна полірування та звітність

### Final Polish

```python
polished = gradient_squeeze(global_best_sol, max_passes=None, quick_mode=False, dyn_bonus=best_cost × 0.001)
```

Необмежений повний local search без `quick_mode` — перевіряє кожну трубу в обидва боки до абсолютної збіжності.

### Експорт (`plot.py`)

| Файл                      | Зміст                                                          |
| ------------------------- | -------------------------------------------------------------- |
| `solution_champion.csv`   | Pipe ID, діаметр, довжина, вартість кожної труби               |
| `optimized_network.inp`   | EPANET-файл з оптимальними діаметрами (для подальшого аналізу) |
| `engineering_report.txt`  | Тиск у кожному вузлі, швидкість і втрати у кожній трубі        |
| `convergence.png`         | Крива збіжності: cost(M$) vs симуляцій                         |
| `convergence_history.csv` | Числові дані для convergence.png                               |
| `network_map.png`         | Кольорова карта топології (ширина ліній ∝ діаметр)             |

---

## 12. Налаштування параметрів

### CLI параметри

| Аргумент     | За замовчуванням            | Опис                                             |
| ------------ | --------------------------- | ------------------------------------------------ |
| `--inp`      | `InputData/Hanoi/Hanoi.inp` | Шлях до EPANET `.inp` файлу                      |
| `--costs`    | `InputData/Hanoi/costs.csv` | Таблиця діаметрів і вартостей                    |
| `--hmin`     | `30.0`                      | Мінімальний тиск (м вод. ст.)                    |
| `--units`    | `mm`                        | `mm` або `in` (дюйми)                            |
| `--cores`    | `0` (всі)                   | Кількість процесів (Островів)                    |
| `--runs`     | `1`                         | Незалежних запусків                              |
| `--run_mode` | `analytical`                | `analytical` або `fast_analytical`               |
| `--v_opt`    | `1.0`                       | Ціл. швидкість потоку (м/с) для velocity-seeding |
| `--max_sims` | `None` (∞)                  | Глобальний бюджет симуляцій                      |
| `--config`   | `None`                      | JSON-файл (перезаписує CLI аргументи)            |

### JSON конфігурація

```json
{
  "inp": "InputData/Balerma/Balerma.inp",
  "costs": "InputData/Balerma/costs.csv",
  "hmin": 20.0,
  "units": "mm",
  "cores": 8,
  "runs": 3,
  "v_opt": 1.2,
  "max_sims": 5000000
}
```

### Класи мереж та автоналаштування

| Клас   | Умова          | Beam Width | SINGLE_CANDIDATES | Поведінка                  |
| ------ | -------------- | ---------- | ----------------- | -------------------------- |
| SMALL  | n < 50         | 5          | n//5              | Повний пошук, всі комбо    |
| MEDIUM | 50 ≤ n < 200   | 5          | n//10             | Стандарт                   |
| LARGE  | 200 ≤ n < 1000 | 8          | n//20             | Focus на high_impact pipes |
| XLARGE | n ≥ 1000       | 8          | 10                | Мінімальні кандидати       |

### Ключові внутрішні константи

| Параметр              | Значення                                 | Де задається                | Роль                                |
| --------------------- | ---------------------------------------- | --------------------------- | ----------------------------------- |
| `BASE_SIM_BUDGET`     | SMALL:1M MEDIUM:3M LARGE:1.5M XLARGE:30M | `AnalyticalSolver.__init__` | Бюджет якщо `max_sims=None`         |
| `stag_limit`          | 4 → 12                                   | адаптивно                   | Поріг стагнації перед T-зростанням  |
| `min_rel_improvement` | 0.0003                                   | `gradient_squeeze`          | Рання зупинка LS                    |
| `water_level`         | best×(1+0.03+0.07T)                      | `_apply_kick`               | Вхід нових рішень у пул             |
| `explosion_threshold` | best×(1.5+0.5T)                          | `_apply_kick`               | Відкидання вибухів                  |
| `basin_tabu maxlen`   | 500                                      | `pool.py` (deque)           | FIFO-пам'ять відвіданих басейнів    |
| `tabu tenure`         | 80 раундів                               | `is_tabu`                   | Скільки раундів рішення tabu        |
| `ipc_immunity`        | 30 (старт) / 50 (після rescue)           | `IslandWorker`              | Захист від надто ранньої міграції   |
| `migration_interval`  | max_sims/(20…50)                         | адаптивно                   | Частота міжостровного обміну        |
| `LRU maxsize`         | 50 000                                   | `SolverContext`             | Для великих мереж збільшити до 200k |

### Рекомендовані значення для тестових мереж

| Мережа  | n труб | Клас  | `--max_sims` | `--cores` | Очікуваний результат             |
| ------- | ------ | ----- | ------------ | --------- | -------------------------------- |
| Hanoi   | 34     | SMALL | 1M           | 5         | ~6.08 M$ (відомий оптимум ~6.08) |
| Balerma | 454    | LARGE | 15M          | 5+        | ~1.93–1.96 M$                    |

### Виведення результатів

```
OutputDataExperiments/
└── 2026-04-14_10-51-09/
    ├── logs/
    │   ├── run_2026-04-14_10-51-09.txt     ← головний лог
    │   └── worker_01.txt                   ← детальний лог воркера
    ├── plots/
    │   ├── convergence.png
    │   └── network_map.png
    └── tables/
        ├── solution_champion.csv
        ├── optimized_network.inp
        ├── engineering_report.txt
        ├── convergence_history.csv
        └── runs_summary.csv
```

### Формат `costs.csv`

```csv
diameter,cost_per_meter
100,45.72
150,70.40
200,98.39
250,129.33
300,180.50
```

Перший стовпець — діаметр у мм (або дюймах якщо `--units in`).  
Другий стовпець — вартість прокладання за метр (будь-яка єдина валюта).
