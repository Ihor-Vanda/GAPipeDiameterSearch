# Water Distribution Network — Документація оптимізатора діаметрів

> **Платформа:** Python 3.10+, Windows/Linux  
> **Метод:** Iterated Local Search + UCB1-адаптація + Island Model паралелізм  
> **Задача:** мінімізація капітальної вартості мережі водопостачання при виконанні обмежень тиску

---

## Зміст

1. [Структура проекту](#1-структура-проекту)
2. [Модель задачі та формули](#2-модель-задачі-та-формули)
3. [Шар симуляції — EPANET C-API](#3-шар-симуляції--epanet-c-api)
4. [SolverContext — спільна структура даних воркера](#4-solvercontext--спільна-структура-даних-воркера)
5. [SolutionPool та tabu-пам'ять](#5-solutionpool-та-tabu-память)
6. [Генерація початкових рішень — SeedFactory](#6-генерація-початкових-рішень--seedfactory)
7. [Локальний пошук — LocalSearch](#7-локальний-пошук--localsearch)
8. [Головний цикл оптимізації — IslandWorker](#8-головний-цикл-оптимізації--islandworker)
9. [Диверсифікація — KickStrategies](#9-диверсифікація--kickstrategies)
10. [Самонавчання параметрів кіків (μ-Learning)](#10-самонавчання-параметрів-кіків-μ-learning)
11. [Паралелізм та Island Model](#11-паралелізм-та-island-model)
12. [Режим fast_analytical](#12-режим-fast_analytical)
13. [Звітність та візуалізація](#13-звітність-та-візуалізація)
14. [Налаштування та запуск](#14-налаштування-та-запуск)

---

## 1. Структура проекту

```
GAPipeDiameterSearch/
├── main.py                      ← CLI-точка входу
├── gui.py                       ← GUI (CustomTkinter)
├── ga_config.py                 ← GAConfig dataclass
├── ga_data.py                   ← load_config (читання costs.csv)
├── ga_utils.py                  ← silence_warnings, format_time, DualLogger
├── water_sim.py                 ← WaterSimulator (EPANET C-API обгортка)
├── plot.py                      ← export_solution, plot_convergence, plot_network_map
├── analytical_solver.py         ← монолітна копія (legacy, для сумісності)
│
└── analytical_solver/           ← модульний пакет (активна версія)
    ├── __init__.py              ← re-export AnalyticalSolver
    ├── orchestrator.py          ← SeedFactory, IslandWorker, AnalyticalSolver
    ├── context.py               ← SolverContext
    ├── local_search.py          ← LocalSearch
    ├── kicks.py                 ← KickStrategies
    ├── pool.py                  ← SolutionPool
    ├── cache.py                 ← LRUCache
    └── fast_math.py             ← Numba JIT функції
```

### Граф залежностей між модулями

```
main.py / gui.py
    │
    ├── GAConfig (ga_config.py) ──► load_config (ga_data.py)
    ├── WaterSimulator (water_sim.py) ──► EPANET C-API (ENepanet)
    │
    └── AnalyticalSolver (orchestrator.py)
            ├── SolverContext (context.py)
            │       ├── LRUCache (cache.py)
            │       └── fast_dijkstra (fast_math.py, Numba JIT)
            ├── SeedFactory (orchestrator.py)
            ├── SolutionPool (pool.py)
            │       └── fast_hamming_distance (fast_math.py, Numba JIT)
            ├── LocalSearch (local_search.py)
            └── KickStrategies (kicks.py)
```

---

## 2. Модель задачі та формули

### Змінні рішення

Мережа містить `n` труб. Кожна труба `i` отримує діаметр із дискретного каталогу:

```
Каталог: {D₀, D₁, ..., D_K}   де D_j < D_{j+1}, одиниці — метри СІ
Змінна:  x_i ∈ {0, 1, ..., K}  (цілочисельний індекс)
Рішення: x = [x₀, x₁, ..., x_{n-1}]
```

### Цільова функція — капітальна вартість

```
C(x) = Σᵢ  Lᵢ · cost(D_{x_i})
```

де:

- `Lᵢ` — довжина труби `i` (метри, з `water_sim.py → wn.get_link(p).length`)
- `cost(D)` — вартість прокладання за метр для діаметру `D` (з `costs.csv`)

### Обмеження

```
p_j(x) ≥ h_min   для всіх вузлів-споживачів j
```

де `p_j` — тиск (м вод. ст.), `h_min` — задається `--hmin`.

### Penalized score (всередині алгоритму)

```
score(x) = C(x) − p_surplus(x) · dyn_bonus
```

де:

- `p_surplus = min_j(p_j) − h_min` — мінімальний надлишок тиску
- `dyn_bonus = min(run_best, global_best) · 0.001 · U(0.95, 1.05)` — стохастичний ваговий коефіцієнт

Знак «мінус» означає: рішення з більшим надлишком тиску (при рівній вартості) отримує **кращий** score. Це запобігає відкиданню рішень, що перевищують `h_min` із запасом — вони більш стабільні.

### Функція штрафу (лише для `evaluate()` — legacy GA mode)

```
f(x) = C(x) + penalty_factor · Σⱼ max(0, h_min − pⱼ(x))
```

Використовується лише в `WaterSimulator.evaluate()` для зворотньої сумісності з GA-режимом.

### Relaxed-feasibility (epsilon window у beam search)

```
is_strictly_valid  = p_surplus ≥ 0.0
is_epsilon_valid   = p_surplus ≥ −0.5 м

score_epsilon = C(x) + |p_surplus| · dyn_bonus · 50.0
```

Рішення з легким порушенням можуть потрапити у пул, але не оголошуються рекордом.

---

## 3. Шар симуляції — EPANET C-API

### Архітектура

`WaterSimulator` є тонкою обгорткою навколо `wntr.epanet.toolkit.ENepanet`. Вся мережа зберігається в C-пам'яті між симуляціями — файли на диск не записуються. Це дає **10–50×** прискорення порівняно зі стандартним `wntr.sim.EpanetSimulator`.

### Ініціалізація (`__init__`)

```
1. wntr.network.WaterNetworkModel(inp_file)
   → топологія (граф, назви труб, довжини у метрах)

2. ENepanet(inp_file, rpt_file, bin_file) → ENopen()
   → відкрити мережу в C-пам'яті

3. ENgetflowunits() → автодетекція системи одиниць:
   flow_units < 5  → US Customary:
     diam_mult = 39.3701  (м → дюйми для C-API)
     press_mult = 0.7032  (PSI → м при читанні)
     hl_mult    = 0.3048  (ft  → м при читанні)
   flow_units ≥ 5  → Metric:
     diam_mult = 1000.0   (м → мм для C-API)
     press_mult = 1.0
     hl_mult    = 1.0

4. Кешування C-індексів для кожної труби та вузла
   (ENgetlinkindex, ENgetnodeindex)
```

### In-memory гідравлічний цикл (`_run_simulation_core`)

```python
ENopenH()           # відкрити hydraulic solver
ENinitH(0)          # 0 = не записувати результати у файл
loop:
    ENrunH()        # розрахунок поточного кроку
    tstep = ENnextH()
    if tstep <= 0: break
ENcloseH()          # вузли зберігають результати останнього кроку в пам'яті
```

### Методи симулятора

| Метод                          | Вхід              | Вихід                             | Призначення         |
| ------------------------------ | ----------------- | --------------------------------- | ------------------- |
| `evaluate(x, pf, eps)`         | індекси x         | `cost + penalty`                  | Legacy GA режим     |
| `get_stats(x)`                 | індекси x         | `(cost, p_min, p_max, crit_node)` | Основна оцінка      |
| `get_heuristics(x)`            | індекси x         | `[hl_i/Lᵢ, ...]` м/м              | Питомі втрати тиску |
| `get_hydraulic_state(diams_m)` | діаметри в метрах | `(flows[], is_feasible)`          | Velocity-seeding    |

**`get_stats`** — найчастіший виклик. Після симуляції ітерує `junction_c_indices`, знаходить `min_p` та відповідний `crit_node`:

```python
for c_idx in self.junction_c_indices:
    p = ENgetnodevalue(c_idx, EN_PRESSURE) * press_si_mult
    if p < min_p: min_p = p; crit_node_idx = c_idx
```

### Захист `__del__`

```python
def __del__(self):
    try:
        if hasattr(self, 'api') and self.api is not None:
            self.api.ENclose()
    except:
        pass
```

`hasattr` перевірка захищає від `AttributeError` якщо `__init__` завершився з помилкою до присвоєння `self.api`.

---

## 4. SolverContext — спільна структура даних воркера

Кожен воркер має свій `SolverContext`, що ізолює весь стан пошуку.

### LRU-кеш симуляцій

```python
sim_cache       = LRUCache(maxsize=50_000)
heuristic_cache = LRUCache(maxsize=50_000)
```

`get_cached_stats(x)` працює так:

```
key = tuple(x)
→ cache hit:  O(1) без EPANET
→ cache miss: sim_count++, get_stats(x), зберегти
```

`LRUCache` реалізований на `collections.OrderedDict`. При переповненні (`.popitem(last=False)`) видаляється найстарший запис.

### CSR-граф для швидкого Dijkstra

При ініціалізації будується Compressed Sparse Row (CSR) представлення:

```python
csr_indptr[i]..csr_indptr[i+1]  → суміжні вузли для вузла i
csr_indices[k]                  → вузол-сусід
csr_edge_pipe[k]                → індекс труби на ребрі (u, v)
```

Ваги рахуються динамічно перед кожним Dijkstra:

```python
csr_weights = 100.0 / (indices_arr[csr_edge_pipe] + 1.0)
```

Менший діаметр → більша вага → Dijkstra знаходить шлях через **найширші** труби. Це і є "домінантний шлях постачання".

### `get_dominant_path(x, crit_node)`

```
target_id = node_to_id[crit_node]
path_ids = fast_dijkstra(source_ids → target_id, weights = 100/(x+1))
→ список індексів труб на домінантному шляху
```

### `is_ghost_solution(x, cost)`

Захист від кешових артефактів. Викликається при поліпшенні > 2%:

```python
cached = sim_cache.get(tuple(x))
if cached: return cached[1] < h_min - 0.01  # перевірка p_min з кешу

# cache miss → реальна симуляція
result = simulator.get_stats(x)
sim_count += 1
return result[1] < h_min - 0.01
```

### Калібрування `_calibrate_simulator`

10 пробних симуляцій після 3 прогрівальних → `sim_speed` (сим/с). Використовується для обчислення `time_limit_sec` коли `max_sims` заданий.

### `baseline_bonus`

```python
avg_cost_diff = mean(|cost[j+1] - cost[j]|)
avg_length    = mean(Lᵢ)
baseline_bonus = avg_cost_diff × avg_length × 0.8
```

Стартовий масштаб для `dyn_bonus`, залежить від конкретного каталогу діаметрів і розмірів труб мережі.

---

## 5. SolutionPool та tabu-пам'ять

### Структура пулу

```python
active_pool       = [(score, cost, sol), ...]  # основний beam пул
tabu_fingerprints = {fingerprint: round_added} # exact-match tabu
kick_tabu_set     = {path_signature, ...}      # tabu для TOPO-INV шляхів
basin_tabu        = deque(maxlen=500)          # FIFO пам'ять басейнів
```

### Точна tabu-перевірка (`is_tabu`)

```python
fingerprint = (int(cost / 50) * 50, tuple(x))  # cost bucket + точний вектор
added_at = tabu_fingerprints.get(fp)
return (current_round - added_at) < tenure=80
```

### Basin signature (`get_basin_signature`)

Огрублений підпис для визначення "вже дослідженого регіону":

```python
if n ≤ 50: return tuple(x)   # точний підпис для малих мереж

chunk_size = n // 25         # ~25 сегментів для великих
sig = [round(mean(x[i:i+chunk_size])) for i in range(0, n, chunk_size)]
return tuple(sig)
```

Два рішення, що відрізняються лише декількома трубами, матимуть однаковий basin signature — вважаються тим же "басейном притягання".

### `basin_tabu` — FIFO через `deque(maxlen=500)`

```python
self.basin_tabu = collections.deque(maxlen=500)

# Додавання: O(1), автоматичне витіснення найстарішого при len=500
self.basin_tabu.append(sig)

# Перевірка:
sig in self.basin_tabu  # O(n) але n ≤ 500, прийнятно
```

### `hamming_distance`

```python
arr1 = np.array(sol1, dtype=np.int32)
arr2 = np.array(sol2, dtype=np.int32)
return fast_hamming_distance(arr1, arr2)   # Numba JIT, O(n)
```

---

## 6. Генерація початкових рішень — SeedFactory

### Velocity-Based Seeding (`make_diverse_seeds`)

**Фізична ідея**: для трубопроводу з потоком Q і цільовою швидкістю v ідеальний діаметр:

```
d_ideal = √(4·|Q| / (π·v))
```

**Ітеративний алгоритм** (конвергує, бо діаметри впливають на потоки):

```
Для v ∈ {1.0, 1.2, 0.8} м/с:
  x ← [max_d_idx × n]   (старт з максимуму)

  Повторювати до 10 разів:
    diams_m = [ctx.diameters[x_i] for i in 0..n]
    flows, _ = simulator.get_hydraulic_state(diams_m)   ← симуляція

    for i in range(n):
      d = sqrt(4·|flows[i]| / (π·v))
      pos = bisect_left(ctx.diameters, d)   ← двійковий пошук у каталозі
      x[i] = min(pos, max_d_idx)

    if x == x_prev: break   ← збіжність

  if infeasible(x): heal_network(x)
  x = gradient_squeeze(x, max_passes=12, quick_mode=True)
  seeds.append(x)
```

### Backbone Seeding (`make_backbone_seed`)

Стратифікація труб за потоком:

```
Відсортувати труби за |Q_i| спадаючи:
  Топ 20% (магістральні):     x_i = max_d_idx
  Решта 30% (транзитні):       x_i = d_ideal(Q_i, v=1.2)  (rounded to catalog)
  Низ 50% (периферійні):       x_i = 0
→ heal_network → gradient_squeeze(max_passes=2)
```

### Reserve Pool (`make_reserve_pool`)

4 рішення з `x_i ~ Uniform(0, max_d_idx)`, зцілені і грубо оптимізовані. Зберігаються у `self.reserve_pool` воркера для `_emergency_pool_diversity`.

### Warm Seeds та система каст (`make_warm_seeds`)

На Epoch > 0 кожному воркеру призначається роль залежно від `worker_id % 4`:

| Роль           | Умова                  | Стратегія                                 | Інтенсивність мутацій |
| -------------- | ---------------------- | ----------------------------------------- | --------------------- |
| **EXPLOITER**  | `adjusted_id % 4 == 0` | ±1 на `n//10` трубах від `archive[rank]`  | Мала                  |
| **RELINKER**   | `adjusted_id % 4 == 1` | Greedy Path-Relinking між двома архівними | Середня               |
| **ARCHITECT**  | `adjusted_id % 4 == 2` | Консенсус топ-3 + ±2 на `n//15` трубах    | Середня               |
| **EXPLORER**   | `adjusted_id % 4 == 3` | ±2 на `n//5` трубах від `archive[rank]`   | Велика                |
| **ADVENTURER** | Останній воркер        | Примусово `make_diverse_seeds()`          | Повна                 |

### Greedy Path-Relinking (`greedy_path_relink`)

```
diff = [i : x_A[i] ≠ x_B[i]]
random.shuffle(diff)       ← стохастичне впорядкування

current = x_A
best = x_A;  best_cost = cost(x_A)

for pipe_idx in diff:
    test = current
    test[pipe_idx] = x_B[pipe_idx]

    if feasible(test):
        current = test
        if cost(test) < best_cost:
            best_cost = cost(test)
            best = test

return best   ← найкраща ПРОМІЖНА точка (не обов'язково x_B!)
```

---

## 7. Локальний пошук — LocalSearch

### `gradient_squeeze` — основний польоровщик

Жадібний покроковий спуск по `score = cost − p_surplus · dyn_bonus`:

```
Ініціалізація:
  dyn_bonus = cost_start × 0.001  (якщо не передано)
  score_best = cost − (p_min − h_min) × dyn_bonus
  milestone_cost = cost;  milestone_pass = 0
  active_indices = [i for i not in locked_pipes]

WHILE improved:
  passes++
  if passes > max_passes: break

  ── Рання зупинка (кожні 3 проходи) ──
  rel_improvement = (milestone_cost − cost) / milestone_cost
  if rel_improvement < min_rel_improvement (0.0003): break

  random.shuffle(active_indices)

  ── Адаптивний поріг фільтрації (quick_mode) ──
  p_surplus = p_min − h_min
  is_critically_tight = p_surplus < 0.1
  loss_threshold = 0.02 (tight) / 0.10 (normal)

  for idx in active_indices:
    if quick_mode AND unit_losses[idx] >= loss_threshold: continue

    ── Спроба DOWNGRADE ──
    if x[idx] > 0:
      test = x; test[idx] -= 1
      c, p, feas = get_cached_stats(test)
      if feas AND p ≥ h_min:
        new_score = c − (p − h_min) × dyn_bonus
        if new_score < best_score: прийняти

    ── Спроба UPGRADE (тільки без quick_mode) ──
    if x[idx] < max_d_idx:
      test = x; test[idx] += 1
      ... аналогічно

return current_x
```

**Режими виклику:**

| Комбінація                 | Де використовується      | Симуляцій на прохід |
| -------------------------- | ------------------------ | ------------------- |
| `quick=True, passes=1`     | Hyperband швидка оцінка  | ~n/5                |
| `quick=True, passes=2-3`   | Після кіків, beam search | ~n/3                |
| `quick=True, passes=12`    | Velocity seeding         | ~n/2                |
| `quick=False, passes=5`    | Beam search top-1        | ~n                  |
| `quick=False, passes=None` | Final Polish             | до збіжності        |

### `heal_network` — відновлення feasibility

Ітеративне виправлення порушень тиску через посилення критичного шляху:

```
WHILE infeasible:
  crit_node = вузол з мінімальним тиском  (з get_cached_stats)
  path_pipes = get_dominant_path(x, crit_node)   ← Dijkstra
  unit_losses = get_cached_heuristics(x)

  candidates = []
  for idx in path_pipes (не в locked_pipes):
    if x[idx] < max_d_idx:
      if n ≥ 200:
        delta_cost = (costs[x[idx]+1] − costs[x[idx]]) × L[idx]
        eff = unit_losses[idx] / (delta_cost × √(max(1, delta_cost)))
      else:
        eff = unit_losses[idx]
      candidates.append((idx, eff))

  best_pipe = argmax(eff)
  x[best_pipe] += 1
  locked_pipes.add(best_pipe)   ← запобігає регресу
  boosts++

if candidates empty: return x, False, boosts
return x, True, boosts
```

**Ключовий момент**: при `n ≥ 200` ефективність нормується на вартість збільшення — алгоритм обирає трубу, що дасть найбільший тиск за найменшу ціну.

### `swap_search` — мікрооптимізація (кожні 8 раундів)

Комбінаторний пошук "обмінів вартості":

```
Отримати lazy_pipes (sorted за unit_loss зростаючи)
Отримати path_pipes до crit_node

if p_surplus > 0.02:   ← не занадто тісно
  for p in lazy_pipes[:down_limit] (unit_loss < 0.05):
    test = x; test[p] -= 1
    if feasible AND better_score: прийняти

if n ≤ 200:   ← тільки для малих мереж
  for up_pipe in path_pipes[-15:]:
    for (d1, d2) in combinations(lazy_pipes[:20], 2):
      test = x; test[up_pipe] += 1; test[d1] -= 1; test[d2] -= 1
      if feasible AND better_score: прийняти
```

Трійки `(up, down₁, down₂)` дають **нейтральні** за вартістю обміни, що можуть покращити score.

### `evaluate_candidate` — атомарна оцінка (для beam mutations)

```python
test = base + upgrade/downgrade на pipes_to_mod
squeezed = gradient_squeeze(test, locked=upgraded_pipes, max_passes=3, quick=True)
cost, p_min = get_cached_stats(squeezed)
if p_surplus < 0: return inf, -inf, None
return cost − p_surplus × dyn_bonus, cost, squeezed
```

### `get_high_impact_pipes` — пріоритизація для LARGE мереж

```python
for i in range(n):
  save_potential = L[i] × (costs[x[i]] − costs[x[i]-1])  # потенційна економія
  risk = max(unit_losses[i], 1e-5)
  impact = save_potential × (1 + risk)
return top_k by impact (descending)
```

---

## 8. Головний цикл оптимізації — IslandWorker

### Стан воркера при запуску

```python
run_best_cost = +∞
run_best_sol  = None
stagnation_counter = 0
stag_limit = 4              # адаптивно зростає до 12
progress_ratio = 0.0        # elapsed / time_budget
is_late_game = False        # True коли progress > 0.5

# μ-learning параметри кіків:
mu_ruin_pct    = 0.05
mu_perturb_pct = 0.10
mu_spatial_pct = 0.20
mu_escape_pct  = 0.20

# UCB1 статистика:
strat_wins = {s: 1.0 for s in all_tracked}
strat_tries = {s: 1.0 for s in all_tracked}
strat_consecutive_fails = {s: 1.0 for s in all_tracked}
```

### Адаптивні параметри за часом

```python
progress_ratio = elapsed / time_budget         # або epoch_sims / max_sims
is_late_game   = progress_ratio > 0.5
stag_limit     = 4 + int(8 × progress_ratio)   # 4 на старті → 12 наприкінці
dyn_bonus      = min(run_best, global_best) × 0.001 × U(0.95, 1.05)
```

### Температура T — "термометр відчаю"

```python
T = min(1.0, stagnation_counter / (stag_limit × 3.0))
```

| T         | Стан               | Наслідки                                  |
| --------- | ------------------ | ----------------------------------------- |
| 0.0 – 0.3 | Активний прогрес   | Холодні стратегії, жорстке прийняття      |
| 0.3 – 0.6 | Помірна стагнація  | Вибір цілі з пулу, розширений water_level |
| 0.6 – 0.9 | Глибока стагнація  | Теплі стратегії, bypass squeeze           |
| 0.9 – 1.0 | Критична стагнація | BASIN_ESCAPE, HAIL MARY, найдальша ціль   |

### Структура одного раунду

```
╔══════════════════════════════════════════════════════════════════╗
║  РАУНД round_idx                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. IPC SYNC    → _process_ipc()      читати стан peer воркерів   ║
║ 2. RESCUE      → _check_rescue()     відстаємо > 2% від global?  ║
║ 3. PUBLISH     → shared_progress[id] = {round, sims, best_cost}  ║
║ 4. PROGRESS    → оновити ratio, late_game, stag_limit, dyn_bonus ║
║ 5. MINI-RESTART→ _check_mini_restart() якщо дуже довга стагнація ║
║ 6. TABU CLEAR  → kick_tabu_set.clear() кожні 6 раундів           ║
║ 7. SWAP        → _apply_swap()       кожні 8 раундів             ║
║ 8. KICK        → _apply_kick()       якщо stagnation ≥ 2         ║
║ 9. FLUSH FLAG  → _force_flush_next?                              ║
║ 10. MUTATE     → _generate_mutations()                           ║
║ 11. BEAM       → _beam_search_and_update()                       ║
║ 12. MIGRATE    → _spatial_crossover() кожні migration_interval   ║
╚══════════════════════════════════════════════════════════════════╝
```

### `_generate_mutations` — beam-мутації

```python
for (score, cost, parent_sol) in active_pool:
  unit_losses = get_cached_heuristics(parent_sol)
  high_friction = sorted(pipes, by unit_loss, desc)  # апгрейди
  low_friction  = sorted(pipes, by unit_loss, asc)   # даунгрейди

  # Адаптивні ліміти за p_surplus:
  if p_surplus > 10: down_limit=15, up_limit=SINGLE_CANDIDATES
  elif p_surplus < 2: down_limit=3,  up_limit=SINGLE_CANDIDATES//2
  else:               down_limit=8,  up_limit=SINGLE_CANDIDATES

  # LARGE/XLARGE: обмежити кандидатів топ-K за impact
  if network_class in (LARGE, XLARGE):
    focus = get_high_impact_pipes(parent_sol, top_k=n//5)
    фільтрувати high_friction і low_friction через focus

  for pipe in high_friction[:up_limit]:   → evaluate_candidate(upgrade)
  for pipe in low_friction[:down_limit]:  → evaluate_candidate(downgrade)
  for (p1,p2) in combinations(high[:5], 2): → evaluate_candidate(upgrade)
```

`SINGLE_CANDIDATES` = `{SMALL:n//5, MEDIUM:n//10, LARGE:n//20, XLARGE:10}`

### `_beam_search_and_update` — відбір та оновлення

```python
min_dist = max(1, (n × 0.05) × (1 − progress_ratio)³)
  ← мінімальна Hamming відстань між рішеннями в пулі
  ← зменшується до нуля наприкінці (конвергенція дозволяється)

dynamic_beam = max(3, BEAM_WIDTH × (1 + 0.5 × (1 − progress_ratio)))
  ← пул ширший на початку, звужується наприкінці

for rank, (score, cost, sol) in enumerate(next_gen_sorted):
  if is_tabu(sol, cost): continue

  # Диференційоване уточнення:
  if   rank == 0:    gradient_squeeze(max_passes=5, quick=not(round%2==0))
  elif rank < 3:     gradient_squeeze(max_passes=3, quick=True)
  else:              без уточнення

  # Перевірка різноманітності:
  for peer in unique_next_pool:
    if hamming(sol, peer) < min_dist: відкинути

  # Ghost Shield:
  if cost < run_best AND improvement > 2%:
    is_ghost = is_ghost_solution(sol, cost)
    if is_ghost: continue

  # Оновлення рекорду:
  if cost < run_best:
    diff = run_best − cost
    if diff > run_best × 0.005: found_new_record = True; stagnation = 0
    else:                        stagnation = max(0, stagnation − 2)  # мікрокрок

active_pool = unique_next_pool[:dynamic_beam]
if found_new_record: stagnation = 0; kick_tabu.clear()
else:                stagnation += 1
```

### `_emergency_pool_diversity`

```python
avg_dist = fast_avg_hamming(pool_matrix)
diversity_threshold = (n // 8) × (1 − progress_ratio)

if avg_dist < threshold AND reserve_pool not empty:
    замінити останній елемент пулу на reserve seed
```

### `_check_mini_restart`

```python
multiplier = {LARGE: 6, інші: 10} // 2 якщо is_late_game
if stagnation_counter ≥ stag_limit × multiplier:
    очистити tabu + pool
    згенерувати 4 мутанти від global_best:
        n_perturb = min(30..45, n//5)
        for each: heal → squeeze → додати в пул
    stagnation = 0
```

---

## 9. Диверсифікація — KickStrategies

### UCB1 вибір стратегії

```python
exploration_C = 0.15 + 0.25 × T   # T=0 → 0.15; T=1 → 0.40

ucb1_score(s) = wins[s]/tries[s]  +  C × √(ln(n_total) / tries[s])
              ← exploitation        ← exploration

strategy = argmax ucb1_score(s) серед valid_strats
```

**Пул стратегій за T:**

| T             | active_pool_strats                                      |
| ------------- | ------------------------------------------------------- |
| T < 0.5       | SHOCK, BOTTLENECK, LOOP_BALANCE, ZERO_SUM, TRIM         |
| 0.5 ≤ T < 0.9 | SPATIAL_PERTURB, SMART_PERTURB, RUIN_RECREATE, TOPO_INV |
| T ≥ 0.9       | BASIN_ESCAPE, SPATIAL_PERTURB, RUIN_RECREATE            |

**Consecutive fails захист**: при `n_fails ≥ max_fails` (3 при T<0.5, 5 при T≥0.5) — стратегія виключається з вибору до першого успіху.

**Вибір цілі для кіку:**

```python
T ≥ 0.8: max Hamming від run_best серед feasible рішень у пулі
T ≥ 0.4: random.choice(active_pool)
T < 0.4: round % 3:
  0 → run_best_sol
  1 → active_pool[0]
  2 → random.choice(active_pool)
```

### Pipeline прийняття kicked-рішення

```
КРОК 1 — РАННІЙ ВІДСІВ:
  raw_deficit = max(0, h_min − p_raw)
  catastrophic_limit = max(20, h_min × (1 + T))
  if raw_deficit > catastrophic_limit: DISCARD

КРОК 2 — HEAL:
  max_allowed_deficit = 0.5 × T   ← relaxed h_min
  if not feas OR p < h_min − max_allowed_deficit:
    heal_locks = set()   для SPATIAL/SMART/RUIN
    heal_locks = locked  для решти
    healed, ok = heal_network(sol, heal_locks)
    if not ok: DISCARD

КРОК 3 — SQUEEZE (не для BASIN_ESCAPE):
  quick_passes = max(1, 2 − int(T × 2))   # T=0→2, T=0.5→1, T=1→1
  quick_sol = gradient_squeeze(passes=quick_passes, quick=True)

  HYPERBAND оцінка:
    hb_margin = 0.03 − 0.02 × progress_ratio   # 3% → 1%
    is_promising = feasible AND quick_cost < run_best × (1 + hb_margin)

    if promising:
      gap = (quick_cost − run_best) / run_best
      deep_passes:
        T ≥ 0.6 → 1
        T ≥ 0.4 → 2
        gap < −0.01% → 8 (LARGE) або 5 (SMALL)
        gap ≤ 0.2% → 3
        gap ≤ 2%  → 2
        інше      → 1

      CONSENSUS FREEZE (T < 0.2, is_late_game, archive ≥ 3):
        знайти труби однакові у топ-3 архіві
        заморозити до freeze_pct=25% від них
        (вимкнути якщо progress > 0.88)

      final = gradient_squeeze(quick_sol, locked=consensus∪locked, passes=deep_passes)

КРОК 4 — ПРИЙНЯТТЯ:
  deficit = max(0, h_min − p_final)
  effective_cost = cost + deficit × (run_best × 0.10)
  explosion_threshold = run_best × (1.5 + 0.5 × T)

  if effective_cost > explosion_threshold: REJECT (Hard)

  water_level = run_best × (1 + 0.03 + 0.07 × T)   # 3% → 10%

  if effective_cost < run_best → Direct Record Update
  elif BASIN_ESCAPE AND deficit=0 → Pool Flush
  elif effective_cost < water_level AND not basin_tabu → Add to Pool
  elif T ≥ 0.95 → HAIL MARY (прийняти примусово)
  else → Reject
```

### Детальні стратегії (формули параметрів)

#### SHOCK (`forcing_hand_kick`)

```
aggressiveness = 0.05 + 0.35 × T
limit = max(1, len(path_pipes) × aggressiveness)
Збільшити limit труб з найвищим unit_loss на critical path
```

#### BOTTLENECK (`upstream_bottleneck_kick`)

```
Фаза 1 — Taper Detection (пройти path_pipes у зворотньому напрямку):
  if x[curr] < x[prev]: збільшити curr → return

Фаза 2 — Fallback:
  search_depth = len(path) × (0.3 + 0.4 × T)
  boost_pct = 0.05 + 0.15 × T
  Підняти boost_pct% найгірших труб у search_depth

Pipe tenure = 10 раундів (failed_pipes tabu)
```

#### TOPO_INV (`topological_inversion_kick`)

```
Знайти до 5 альтернативних шляхів до crit_node
  (множачи ваги відвіданих ребер × 5 після кожного SP)
Відфільтрувати tabu-підписи
Вибрати шлях з мінімальним overlap з домінантним

target_capacity_idx = mean(x[i] for i in dom_pipes)
aggressiveness = 0.10 + 0.50 × T
max_pipes = len(alt_path) × aggressiveness
Підняти chosen pipes до min(target_capacity_idx, max_d_idx)
```

#### LOOP_BALANCE (`loop_balancing_kick`)

```
cycles = nx.cycle_basis(G)   (кешується в self._cached_cycles)
restrict_pct = 0.10 + 0.30 × T
max_allowed_drop = 1 + int(2 × T)

Для кожного циклу (до 5 кандидатів):
  chosen = random.sample(cycle_pipes, n_restrict)
  for drop in range(max_drop, 0, -1):
    kicked = x; kicked[chosen] -= drop
    healed, ok = heal_network(kicked, locked=chosen)
    if ok: squeeze → record as candidate

Повернути random.choice(top-3 за вартістю)
Pipe tenure = min(80, n // 5) раундів
```

#### ZERO_SUM (`zero_sum_shift_kick`)

```
upgrades[i] = (i, cost_invest, unit_loss[i] / cost_invest)
  cost_invest = L[i] × (costs[x[i]+1] − costs[x[i]])

downgrades[i] = (i, cost_save, cost_save / unit_loss[i])
  cost_save = L[i] × (costs[x[i]] − costs[x[i]-1])

Виключити труби з zero_sum_tabu (tenure 15 раундів)
search_pool_size = max(15, n // 10)
tests_limit = 5 (T<0.3) або 2 (T≥0.3)
max_downgrades = n//50 × {1, 3, 6} залежно від T

for up in valid_upgrades[:search_pool_size]:
  накопичувати downgrades поки savings > cost_invest
  heal → evaluate → зберегти як кандидат
```

#### TRIM (`peripheral_trim_kick`)

```
periphery = труби поза critical path з x[i] ≤ 0.6 × max_d_idx
  сортовані за unit_loss зростаючи

trim_pct = 0.02 + 0.13 × T
pipes_to_cut = max(1, len(periphery) × trim_pct)

25 спроб: random.sample → heal → evaluate
Повернути random.choice(top-3)
```

#### SMART_PERTURB (`smart_perturbation_kick`)

```
μ = mu_perturb_pct + 0.15 × T
σ = 0.02 + 0.05 × T
target_pct ~ Gauss(μ, σ), clip [0.01, 0.30]

n_perturb = min(max_perturb, n × target_pct)
  max_perturb = {SMALL:∞, MEDIUM:25, LARGE:40, XLARGE:60}

Вибрати труби з 0 < x < max_d (або всі якщо замало)
if T > 0.6: delta ~ {-2,-1,+1,+2} weights=[1,3,3,1]
else:        delta ~ {-1, +1}

heal → return (healed, locked, msg, target_pct)
```

#### RUIN_AND_RECREATE (`ruin_and_recreate_kick`)

```
ruin_center = crit_node (T < 0.8) або random node (T ≥ 0.8)

μ = mu_ruin_pct + 0.10 × T
σ = 0.01 + 0.04 × T
target_pct ~ Gauss(μ, σ), clip [0.01, 0.25]
target_pipes = max(3, n × target_pct)

cutoff = {n<50:3, n<200:4, n<1000:6, n≥1000:8}
BFS від ruin_center з cutoff → cluster_pipes (closest first)

max_catalog_drop = max(1, max_d_idx // 3)
max_drop = min(1 + int(2 × T), max_catalog_drop)

for p in cluster_pipes:
  x[p] = max(0, x[p] − random.randint(1, max_drop))

heal(kicked, locked=set())   ← locked=set() для максимальної гнучкості
return (healed, set(), msg, target_pct)
```

#### BASIN_ESCAPE (`basin_escape`)

```
Знайти diverse_sol з global_archive з max Hamming до indices
Потрібно best_dist ≥ max(2, n × 0.03)

μ = mu_escape_pct + 0.30 × T
σ = 0.05 + 0.05 × T
target_pct ~ Gauss(μ, σ), clip [0.05, 0.60]

n_replace = len(diff_pipes) × target_pct
Замінити n_replace труб значеннями з diverse_sol

heal(kicked, set())
basin_tabu.append(signature(run_best_sol))   ← поточний басейн ← tabu

if accepted:
  active_pool.clear()
  _force_flush_next = True
  ipc_immunity = 50
return (healed, set(), msg, target_pct)
```

#### SPATIAL_PERTURB (`spatial_perturb_kick`)

```
base_radius = {n<50:2, n<200:3, n<1000:5, n≥1000:8}
radius = base_radius + int(base_radius × T)

epicenter = random.choice(G.nodes())
local_nodes = BFS(epicenter, cutoff=radius)
local_pipes = всі труби суміжні з local_nodes

locked = {i : i not in local_pipes}   ← заморозити все поза патчем

μ = mu_spatial_pct + 0.15 × T   ← НЕЗАЛЕЖНИЙ від mu_perturb!
σ = 0.02 + 0.05 × T
target_pct ~ Gauss(μ, σ), clip [0.01, 0.40]
target_mutations = min(max_perturb, len(local_pipes) × target_pct)

Мутувати target_mutations труб у local_pipes
Fallback → smart_perturbation_kick якщо local_pipes порожній

return (kicked, locked, msg, target_pct)
```

---

## 10. Самонавчання параметрів кіків (μ-Learning)

### Exponential Moving Average (EMA)

```python
μ_new = 0.9 × μ_old + 0.1 × used_pct
```

де `used_pct` — фактична частка труб, задіяних кіком (повертається як `res[3]`).

### Параметри та умови оновлення

| Параметр         | Стратегія       | μ₀   | Умова оновлення                               |
| ---------------- | --------------- | ---- | --------------------------------------------- |
| `mu_ruin_pct`    | RUIN_RECREATE   | 0.05 | `deficit == 0 AND cost < run_best`            |
| `mu_perturb_pct` | SMART_PERTURB   | 0.10 | `deficit == 0 AND cost < run_best`            |
| `mu_spatial_pct` | SPATIAL_PERTURB | 0.20 | `deficit == 0 AND cost < run_best`            |
| `mu_escape_pct`  | BASIN_ESCAPE    | 0.20 | `BASIN_ESCAPE accepted AND used_pct not None` |

**Критично**: `mu_perturb_pct` і `mu_spatial_pct` — **різні параметри**. Семантика `target_pct` відрізняється: SMART рахує від `n`, SPATIAL — від `len(local_pipes)`.

### Динамічне sampling навколо μ

```
dynamic_mu = μ + slope × T      # μ зростає зі стагнацією
dynamic_sigma = σ₀ + σ_T × T    # дисперсія теж зростає
target_pct = max(lo, min(hi, Gauss(dynamic_mu, dynamic_sigma)))
```

При T → 1 алгоритм автоматично пробує **більші** розміри втручання, навіть якщо μ навчився малим.

### UCB1 reward/decay при успіху

```python
improvement_pct = diff / run_best_cost
reward = 10.0 × improvement_pct × 100   # пропорційно відносному поліпшенню
strat_wins[strategy] += max(1.0, reward)

# Відносна переоцінка всіх стратегій:
for k in strat_wins:
    strat_wins[k] *= 0.97
    strat_tries[k] = max(1.0, strat_tries[k] * 0.97)
```

---

## 11. Паралелізм та Island Model

### Архітектура

```
AnalyticalSolver.solve_standalone()
  │
  ├── multiprocessing.Pool(N workers, initializer=worker_init)
  │       └── worker_init: WaterSimulator у temp dir кожного процесу
  │
  └── для кожного воркера:
        mp_pool.apply_async(worker_task, args=(
            diameters, v_opt, time_budget, global_best, archive,
            seed, worker_id, shared_progress, log_dir, epoch,
            max_sims, n_workers
        ))
```

### Shared Memory (`multiprocessing.Manager().dict()`)

| Ключ                                 | Тип                        | Пише                   | Читає    |
| ------------------------------------ | -------------------------- | ---------------------- | -------- |
| `shared_progress[wid]`               | `{round, sims, best_cost}` | воркер `wid`           | всі      |
| `shared_progress['global_best']`     | `(cost, sol)`              | воркер-переможець      | всі      |
| `shared_progress[f'best_sol_{wid}']` | `(cost, sol)`              | воркер `wid`           | peer IPC |
| `shared_progress['global_archive']`  | `[(cost, sol), ...]`       | оркестратор кожні ~30с | всі      |

### Passive IPC (`_process_ipc`)

```python
for i in range(n_workers) if i != self_id:
  peer = shared_progress[f'best_sol_{i}']

  Умови прийняття peer рішення:
    peer_cost < run_best × 0.995        # краще на 0.5%
    peer_cost < last_injected_cost − 1  # справді нове
    NOT (is_adventurer AND progress < 0.85)  # ADVENTURER чекає до пізньої гри
    NOT (stagnation < stag_limit AND diff < 2%)  # не переривати активний пошук
```

### Rescue (`_check_rescue`)

```python
global_lag = (run_best − global_best) / global_best

if (global_lag > 2% AND stagnation ≥ 2×stag_limit) OR global_lag > 5%:
    run_best = global_best_cost
    pool.insert(global_best_sol with score − 1e6)  # пріоритет у beam
    stagnation = 0; kick_tabu.clear()
```

### Migration — 3-dimensional crossover

```
T_migration = stagnation / (stag_limit × 3)

T < 0.3:   COLD migration
  hybrid = spatial_crossover(run_best, global_best, T=0.25)

  # Додатковий peer noise (3D):
  знайти peer зі cost ≠ global_best (різноманітність)
  if found: hybrid = spatial_crossover(hybrid, peer_sol, T=0.05)

  heal → pool.insert(0, hybrid); stagnation=0; ipc_immunity=50

T ∈ [0.3, 0.7):   WARM migration
  hybrid = spatial_crossover(run_best, global_best, T)
  if hybrid < run_best: прийняти як новий run_best

T ≥ 0.7:   EXPLORATION SHIELD — ігнорувати global_best
```

`spatial_crossover`: epicenter → BFS(radius=2+6T) → `local_pipes` отримують значення від `donor_sol`.

### `_migration_interval_sims` — адаптивна частота

```python
interval = max(1000, (max_sims // 20) × (1 − 0.6 × progress_ratio))
```

Початок: рідко (кожні ~5% бюджету). Кінець: частіше (кожні ~2%).

### Глобальний архів (`_build_diverse_archive`)

Після epoch оркестратор збирає топ-6 різноманітних рішень:

```python
archive = sorted_results[:elite_count=2]   # топ-2 за вартістю

min_diff = max(15, min(45, n × 0.08))      # мінімальна Hamming від архіву

for cost, sol in sorted_results[2:]:
  if min_hamming_to_archive(sol) ≥ min_diff:
    archive.append(...)
  if len(archive) == 6: break
```

---

## 12. Режим `fast_analytical`

Запускається з `--run_mode fast_analytical`. Не використовує Island Model — один потік без кіків.

```
1. make_diverse_seeds_for_fast():
   16 швидкостей {0.5, 0.6, ..., 2.0} × 5 ітерацій velocity-seeding

2. make_backbone_seed(v) для v ∈ {0.8, 1.0, 1.2, 1.5, 1.8}:
   Trunk-Branch стратифікація

3. Відфільтрувати feasible seeds (дедублікувати)

4. Сортувати за вартістю; взяти top-5

5. Для кожного з top-5:
   polished = gradient_squeeze(raw_sol, max_passes=None, quick=False)
   Якщо кращий → глобальний рекорд
```

Час: секунди – хвилини. Якість: хороша стартова точка, не конкурує з `analytical` за якістю.

---

## 13. Звітність та візуалізація

### `export_solution`

```python
# Читати оптимальні діаметри і записати:
1. solution_champion.csv
   Pipe ID | Start Node | End Node | Diameter (mm) | Length | Cost

2. optimized_network.inp
   EPANET файл з оптимальними діаметрами (wntr.network.write_inpfile)

3. engineering_report.txt (через wntr.sim.EpanetSimulator):
   - Вартість / Час / Кількість симуляцій
   - Тиск у кожному вузлі (sorted by pressure asc)
   - Швидкість і втрати у кожній трубі (sorted by velocity desc)
```

### `plot_convergence`

Крива збіжності зі step-функцією (монотонно спадна):

```python
plt.step(evals, costs_M, where='post')
# Також зберігає convergence_history.csv
```

### `plot_network_map`

Кольорова теплова карта топології:

```python
edge_color = real_diams_m        # колір по діаметру (viridis)
line_widths = 1 + 4 × normalized_diam   # ширина ∝ діаметр
# Вузли: чорні (junction), сині квадрати (reservoir), червоні трикутники (tank)
```

### Дерево виведення

```
OutputDataExperiments/
└── YYYY-MM-DD_HH-MM-SS/
    ├── logs/
    │   ├── run_YYYY-MM-DD_HH-MM-SS.txt    ← stdout + stderr (DualLogger)
    │   └── worker_01.txt ... worker_N.txt  ← детальні логи воркерів
    ├── plots/
    │   ├── convergence.png
    │   └── network_map.png
    └── tables/
        ├── solution_champion.csv
        ├── optimized_network.inp
        ├── engineering_report.txt
        ├── convergence_history.csv
        └── runs_summary.csv    (якщо --runs > 1)
```

---

## 14. Налаштування та запуск

### CLI параметри

| Аргумент     | За замовч. | Тип    | Опис                                   |
| ------------ | ---------- | ------ | -------------------------------------- |
| `--inp`      | Hanoi.inp  | str    | Шлях до EPANET `.inp`                  |
| `--costs`    | costs.csv  | str    | Таблиця діаметрів і вартостей          |
| `--hmin`     | 30.0       | float  | Мінімальний тиск (м вод. ст.)          |
| `--units`    | mm         | mm\|in | Одиниці діаметрів у costs.csv          |
| `--cores`    | 0 (всі)    | int    | Кількість островів                     |
| `--runs`     | 1          | int    | Незалежних запусків                    |
| `--run_mode` | analytical | str    | `analytical` або `fast_analytical`     |
| `--v_opt`    | 1.0        | float  | Цільова швидкість для velocity-seeding |
| `--max_sims` | None (∞)   | int    | Глобальний бюджет симуляцій            |
| `--config`   | None       | str    | JSON (перезаписує CLI аргументи)       |

### JSON конфіг (рекомендовано для повторних запусків)

```json
{
  "inp": "InputData/Balerma/Balerma.inp",
  "costs": "InputData/Balerma/costs.csv",
  "hmin": 20.0,
  "units": "mm",
  "cores": 8,
  "runs": 3,
  "v_opt": 1.2,
  "max_sims": 15000000
}
```

```bash
python main.py --config balerma_15M.json
```

### Класи мереж та автоналаштування

| Клас   | n труб  | beam_width | SINGLE_CANDIDATES | Особливості                           |
| ------ | ------- | ---------- | ----------------- | ------------------------------------- |
| SMALL  | < 50    | 5          | n//5              | Повний swap, точна basin sig          |
| MEDIUM | 50–199  | 5          | n//10             | Стандарт                              |
| LARGE  | 200–999 | 8          | n//20             | focus на high_impact, обмежений SMART |
| XLARGE | ≥ 1000  | 8          | 10                | Мінімальні кандидати, великі radii    |

### Ключові внутрішні константи

| Параметр              | Значення                                    | Де задається                |
| --------------------- | ------------------------------------------- | --------------------------- |
| `BASE_SIM_BUDGET`     | SMALL:1M, MEDIUM:3M, LARGE:1.5M, XLARGE:30M | `AnalyticalSolver.__init__` |
| `stag_limit`          | 4 → 12                                      | Адаптивно в `run()`         |
| `water_level`         | `best × (1 + 0.03 + 0.07T)`                 | `_apply_kick`               |
| `explosion_threshold` | `best × (1.5 + 0.5T)`                       | `_apply_kick`               |
| `min_rel_improvement` | 0.0003                                      | `gradient_squeeze`          |
| `basin_tabu maxlen`   | 500                                         | `pool.py` (deque)           |
| `tabu tenure`         | 80 раундів                                  | `is_tabu`                   |
| `ipc_immunity`        | 30 / 50 (після rescue)                      | `IslandWorker`              |
| `LRU cache maxsize`   | 50 000                                      | `SolverContext`             |

### Рекомендовані налаштування за мережею

| Мережа  | n   | Клас  | `--max_sims` | `--cores` | Цільовий результат |
| ------- | --- | ----- | ------------ | --------- | ------------------ |
| Hanoi   | 34  | SMALL | 1M           | 5         | ~6.08 M$           |
| Balerma | 454 | LARGE | 15M          | 5+        | ~1.93–1.96 M$      |

### Формат `costs.csv`

```csv
diameter,cost_per_meter
100,45.72
150,70.40
200,98.39
250,129.33
300,180.50
```

Перший стовпець — діаметр (мм або дюйми залежно від `--units`).  
Другий стовпець — вартість за метр (будь-яка валюта, але єдина для всіх рядків).

Підтримуються варіанти назв стовпців: `Diameter/Diam/D/diameter/size` і `Cost/cost/Price/price/UnitCost`.
