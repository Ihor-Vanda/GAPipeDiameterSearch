"""`PROJECT_MAP.md` не має посилатися на те, чого в коді немає.

Мапу читає не людина, а ШІ-агент, і читає її **замість** коду — саме для
цього вона існує. Тому найдорожча її помилка не неповнота, а впевнене
посилання на те, чого вже немає. Агент іде за якорем `context.py:220`,
знаходить там кінець файлу або метод, якого мапа не описує, — і або
витрачає контекст на пошук, або, гірше, будує правку на описі, що
розійшовся з кодом.

Автоматично перевіряється те, що автоматизується без запуску солвера:

1. кожен якір `файл.py:рядок` вказує на існуючий файл і на рядок у його
   межах;
2. кожен символ покажчика (§16) існує у файлі, який там названо;
3. кожне `Клас.атрибут` для класу проєкту існує — з урахуванням домішок
   і базових класів (атрибутом вважається метод, поле класу або
   присвоєння `self.x`);
4. кожне ім'я ВЕЛИКИМИ ЛІТЕРАМИ в коді мапи існує в коді — як ім'я або
   як рядкова константа (так проходять імена стратегій, класи мереж і
   маркери логів на кшталт `[TIGHTNESS]`);
5. кожен згаданий `*.py` існує в репозиторії;
6. кожне ім'я, з якого починається код у заголовку картки
   (`#### heal_network(...)`), існує в коді.

Чого тут свідомо **немає** — перевірки, що якір влучає точно у
визначення. Номери рядків зсуваються з кожною правкою вище, і тест, що
падає від будь-якої зміни коду, навчив би його ігнорувати. Мапа сама
каже, що рядки прив'язані до знімка, а надійний якір — ім'я символу;
тому тут суворо перевіряються імена, а рядки — лише на межі файлу.

Код читається через `ast`, без імпорту: інакше перевірка тексту тягнула
б за собою wntr, numba і tkinter.
"""

import ast
import fnmatch
import os
import re
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "PROJECT_MAP.md"

pytestmark = pytest.mark.skipif(not MAP.exists(), reason="немає PROJECT_MAP.md")

#: Код, який описує мапа. `attic/` і `experiments/` вона не описує.
CODE_GLOBS = ("pipeopt/**/*.py", "main.py", "gui.py", "tools/*.py",
              "tests/conftest.py")

#: Каталоги, де не шукаються згадані файли: службові й результати прогонів.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "OutputDataExperiments",
             "_temp_sim_data"}

#: `context.py:220`, `beam.py:62-106`, `water_sim.py:52, 56, 127`.
ANCHOR = re.compile(r"([\w/]+\.py):(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)")

#: Будь-яке ім'я файлу з `.py`. Шаблони на кшталт `kicks_*.py` сюди не
#: потрапляють: зірочка розриває ім'я.
PY_FILE = re.compile(r"(?<![\w/.])([\w/]+\.py)(?![\w])")

#: `SolverContext.score`, `KickArgs.mu_*`. Голова мусить мати малу літеру,
#: щоб `KICK_REGISTRY.get` не вважався класом.
CAMEL_ATTR = re.compile(r"(?<![\w.])([A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*)\.([A-Za-z_][\w*]*)")

#: `SURPLUS_KNEE`, `HOT_*`, `SHOCK`. Після крапки не беруться: `Verdict.POOL`
#: перевіряє `CAMEL_ATTR`, а `self.SINGLE_CANDIDATES` — це поле. Частини
#: імен файлів і каталогів (`RUNNING.md`, `AUDIT_2026-09-21.md`,
#: `EPANETINPUTFILESFOR7NEWORKS/`) — не імена коду, тому відсіюються.
UPPER_NAME = re.compile(
    r"(?<![\w.\-])([A-Z][A-Z0-9]*(?:_(?:[A-Z0-9]+|\*))+|[A-Z][A-Z0-9]{2,})"
    r"(?![\w/])(?!\.(?:md|py|inp|csv|json|txt|png)\b)(?!-\d)")

#: ВЕЛИКІ слова в коді мапи, які посилаються не на Python, а на формат
#: `.inp` або на документацію. Значення — чому слово не є кодом.
NOT_PYTHON = {
    "PIPES": "секція `[PIPES]` файлу `.inp`",
    "AUDIT_*": "шаблон імен файлів ревізії (`AUDIT_2026-09-21.md`)",
}

#: Блоки ```lang ... ```. У блоках `mermaid` підписи — проза («CLI», «CSR»),
#: тому перевірка імен їх пропускає.
FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`([^`\n]+)`")


# --- мапа ----------------------------------------------------------------

@lru_cache(maxsize=None)
def map_text():
    return MAP.read_text(encoding="utf-8")


def line_of(offset):
    return map_text().count("\n", 0, offset) + 1


@lru_cache(maxsize=None)
def code_fragments():
    """Усе, що в мапі записано як код: фрагменти в лапках і блоки.

    Повертає `(зсув у файлі, текст)`. Перевірки імен дивляться лише сюди:
    у прозі ВЕЛИКІ слова — це абревіатури (EPS, IPC), а не посилання на код.
    Блоки замінюються пробілами тієї самої довжини, щоб зсуви фрагментів у
    лапках лишались справжніми.
    """
    text = map_text()
    fragments = [(m.start(2), m.group(2)) for m in FENCE.finditer(text)
                 if m.group(1).strip() != "mermaid"]
    prose = FENCE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    fragments += [(m.start(1), m.group(1)) for m in INLINE_CODE.finditer(prose)]
    return tuple(fragments)


# --- код -----------------------------------------------------------------

@lru_cache(maxsize=None)
def code_files():
    files = set()
    for pattern in CODE_GLOBS:
        files.update(p for p in ROOT.glob(pattern) if "__pycache__" not in p.parts)
    return tuple(sorted(files))


@lru_cache(maxsize=None)
def syntax_tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@lru_cache(maxsize=None)
def repo_python_files():
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        found += [Path(dirpath) / name for name in filenames if name.endswith(".py")]
    return tuple(found)


def resolve(name):
    """Файли, на які може вказувати ім'я з мапи (`beam.py`, `island/loop.py`)."""
    direct = ROOT / name
    if direct.is_file():
        return [direct]
    suffix = "/" + name
    return [p for p in repo_python_files() if p.as_posix().endswith(suffix)]


def _stored_names(target):
    """Імена, які присвоює ціль: `x`, `a, b`, `self.x` → (імена, поля self)."""
    names, self_fields = set(), set()
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == "self"):
            self_fields.add(node.attr)
    return names, self_fields


def _assignment_targets(node):
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return [node.target]
    return []


@lru_cache(maxsize=None)
def module_names(path):
    """Імена верхнього рівня модуля: функції, класи, присвоєння."""
    names = set()
    for node in syntax_tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        for target in _assignment_targets(node):
            names |= _stored_names(target)[0]
    return frozenset(names)


@lru_cache(maxsize=None)
def class_registry():
    """Ім'я класу → список `(власні атрибути, імена базових класів, файл)`.

    Список, а не одне значення: однакові імена класів у різних файлах
    об'єднуються, а не перетирають одне одного.
    """
    registry = {}
    for path in code_files():
        for node in ast.walk(syntax_tree(path)):
            if not isinstance(node, ast.ClassDef):
                continue
            own = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    own.add(item.name)
                for target in _assignment_targets(item):
                    own |= _stored_names(target)[0]
            for item in ast.walk(node):
                for target in _assignment_targets(item):
                    own |= _stored_names(target)[1]
            bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                     for b in node.bases]
            registry.setdefault(node.name, []).append((frozenset(own), bases, path))
    return registry


def attributes_of(class_name, _seen=None):
    """Атрибути класу разом з успадкованими від класів проєкту.

    Домішки — головна причина, чому це потрібно: `IslandWorker` не має
    жодного власного методу пошуку, усі вони в `BeamSearch`,
    `Diversification` та інших.
    """
    seen = _seen if _seen is not None else set()
    if class_name in seen:
        return set()
    seen.add(class_name)
    result = set()
    for own, bases, _ in class_registry().get(class_name, []):
        result |= own
        for base in bases:
            result |= attributes_of(base, seen)
    return result


def classes_in(path):
    return {name for name, entries in class_registry().items()
            if any(p == path for _, _, p in entries)}


@lru_cache(maxsize=None)
def code_vocabulary():
    """Усі імена, які код визначає, і текст усіх його рядкових констант.

    Docstrings до констант не входять: у них згадується багато чого, що
    кодом не є, і перевірка через них пропускала б вигадані імена.
    """
    names, literals = set(), []
    for path in code_files():
        tree = syntax_tree(path)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    docstrings.add(id(first.value))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                names.add(node.attr)
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and id(node) not in docstrings):
                literals.append(node.value)
    return frozenset(names), "\n".join(literals)


def matches(pattern, names):
    """Чи є ім'я, що відповідає `pattern`; `*` — будь-який хвіст (`make_*`)."""
    if "*" in pattern:
        return any(fnmatch.fnmatchcase(n, pattern) for n in names)
    return pattern in names


# --- перевірки -----------------------------------------------------------

def anchors():
    return [(line_of(m.start()), m.group(1), m.group(2))
            for m in ANCHOR.finditer(map_text())]


def test_every_anchor_points_inside_an_existing_file():
    problems = []
    for line_no, name, numbers in anchors():
        candidates = resolve(name)
        if len(candidates) != 1:
            problems.append(f"рядок {line_no}: `{name}` — "
                            + ("файлу немає" if not candidates else
                               f"неоднозначно: {[str(p.relative_to(ROOT)) for p in candidates]}"))
            continue
        length = len(candidates[0].read_text(encoding="utf-8").splitlines())
        for number in map(int, re.findall(r"\d+", numbers)):
            if not 1 <= number <= length:
                problems.append(f"рядок {line_no}: `{name}:{number}` — у файлі "
                                f"{length} рядків")

    assert not problems, "якорі ведуть у нікуди:\n  " + "\n  ".join(problems)


def test_every_mentioned_python_file_exists():
    existing = {p.as_posix() for p in repo_python_files()}
    problems = []
    for match in PY_FILE.finditer(map_text()):
        name = match.group(1)
        if not any(p == f"{ROOT.as_posix()}/{name}" or p.endswith("/" + name)
                   for p in existing):
            problems.append(f"рядок {line_of(match.start())}: `{name}`")

    assert not problems, "мапа згадує файли, яких немає:\n  " + "\n  ".join(problems)


def index_rows():
    """Рядки покажчика символів: `(номер, символи, файли)`."""
    text = map_text()
    start = text.index("## 16.")
    rows = []
    for offset, line in _lines_with_offsets(text, start):
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        symbols = INLINE_CODE.findall(cells[0])
        files = [p for name in PY_FILE.findall(cells[1]) for p in resolve(name)]
        rows.append((line_of(offset), symbols, files))
    return rows


def _lines_with_offsets(text, start):
    offset = start
    for line in text[start:].split("\n"):
        yield offset, line
        offset += len(line) + 1


def _resolve_row(symbols, files):
    """Символи рядка, яких немає у файлах рядка.

    Запис рядка стислий: `WaterSimulator.get_stats` / `get_heuristics` —
    друге ім'я належить тому самому класу. Тому голе ім'я шукається і на
    верхньому рівні файлів рядка, і серед атрибутів класу, названого
    раніше в цьому ж рядку. `main.build_parser` — голова тут модуль, а не
    клас.
    """
    stems = {p.stem: p for p in files}
    top_level = set().union(*(module_names(p) for p in files)) if files else set()
    row_classes = set().union(*(classes_in(p) for p in files)) if files else set()
    context = None
    missing = []
    for symbol in symbols:
        head, _, tail = symbol.partition(".")
        if tail and head in stems:
            if not matches(tail, module_names(stems[head])):
                missing.append(symbol)
            continue
        if tail:
            if head not in row_classes or not matches(tail, attributes_of(head)):
                missing.append(symbol)
            context = head
            continue
        in_context = context is not None and matches(head, attributes_of(context))
        if not (matches(head, top_level) or in_context):
            missing.append(symbol)
        if head in row_classes:
            context = head
    return missing


def test_every_indexed_symbol_exists_where_the_index_says():
    problems = []
    for line_no, symbols, files in index_rows():
        if not files:
            problems.append(f"рядок {line_no}: не названо жодного існуючого файлу")
            continue
        missing = _resolve_row(symbols, files)
        if missing:
            where = ", ".join(str(p.relative_to(ROOT)) for p in files)
            problems.append(f"рядок {line_no}: {missing} немає в {where}")

    assert not problems, "покажчик посилається на неіснуючі символи:\n  " + "\n  ".join(problems)


def class_mentions():
    """`Клас.атрибут` для класів проєкту — будь-де в мапі."""
    registry = class_registry()
    return [(line_of(m.start()), m.group(1), m.group(2))
            for m in CAMEL_ATTR.finditer(map_text()) if m.group(1) in registry]


def test_every_class_attribute_the_map_names_exists():
    problems = sorted({f"рядок {line_no}: `{head}.{tail}`"
                       for line_no, head, tail in class_mentions()
                       if not matches(tail, attributes_of(head))})

    assert not problems, ("мапа називає атрибути, яких у класів немає:\n  "
                          + "\n  ".join(problems))


def upper_names():
    return [(line_of(start + m.start(1)), m.group(1))
            for start, fragment in code_fragments()
            for m in UPPER_NAME.finditer(fragment)]


def test_every_constant_or_literal_the_map_names_exists_in_code():
    names, literals = code_vocabulary()
    problems = set()
    for line_no, token in upper_names():
        if token in NOT_PYTHON or matches(token, names):
            continue
        word = re.escape(token).replace(r"\*", r"\w*")
        if re.search(rf"(?<![A-Za-z0-9_]){word}(?![A-Za-z0-9_])", literals):
            continue
        problems.add(f"рядок {line_no}: `{token}`")

    assert not problems, ("у коді мапи є імена, яких немає ні в коді, ні в його "
                          "рядкових константах:\n  " + "\n  ".join(sorted(problems)))


#: Ім'я на початку фрагмента в заголовку: `gradient_squeeze(indices, …)` →
#: `gradient_squeeze`, `SolverContext.score(…)` → `SolverContext.score`.
HEADING_NAME = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)")


def heading_names():
    """Імена, з яких починаються фрагменти коду в заголовках.

    Заголовок картки — це те, що агент шукає першим (`#### heal_network`).
    Голі імена в прозі тут не перевіряються: там вони здебільшого локальні
    змінні (`dyn_bonus`, `real_T`), і перевірка давала б хибні падіння.
    """
    found = []
    for match in re.finditer(r"^#{2,6} .*$", map_text(), flags=re.MULTILINE):
        for fragment in INLINE_CODE.findall(match.group(0)):
            name = HEADING_NAME.match(fragment)
            # Файли й каталоги (`worker.py`, `experiments/`) — не імена коду.
            if name and ".py" not in fragment and "/" not in fragment:
                found.append((line_of(match.start()), name.group(1)))
    return found


def test_every_name_a_heading_starts_with_exists_in_code():
    names, _ = code_vocabulary()
    registry = class_registry()
    problems = []
    for line_no, name in heading_names():
        head, _, tail = name.partition(".")
        if tail and head not in registry:
            continue          # `multiprocessing.Manager` — не клас проєкту
        ok = matches(tail, attributes_of(head)) if tail else name in names
        if not ok:
            problems.append(f"рядок {line_no}: `{name}`")

    assert not problems, ("заголовки карток називають те, чого в коді немає:\n  "
                          + "\n  ".join(problems))


def test_the_map_says_which_snapshot_it_describes():
    """Без коміту знімка читач не знає, наскільки довіряти номерам рядків."""
    assert re.search(r"\*\*Знімок:\*\* commit `[0-9a-f]{7,40}`", map_text())


def test_the_checks_actually_found_something():
    """Самоохорона: перейменований розділ або зламаний вираз дав би
    порожні списки — і всі перевірки вище зеленіли б завжди."""
    found = {
        "якорів": len(anchors()),
        "рядків покажчика": len(index_rows()),
        "символів у покажчику": sum(len(s) for _, s, _ in index_rows()),
        "згадок Клас.атрибут": len(class_mentions()),
        "імен великими літерами": len(upper_names()),
        "імен у заголовках": len(heading_names()),
        "класів у коді": len(class_registry()),
    }
    minimum = {"якорів": 200, "рядків покажчика": 50, "символів у покажчику": 100,
               "згадок Клас.атрибут": 30, "імен великими літерами": 100,
               "імен у заголовках": 20, "класів у коді": 30}
    short = {k: v for k, v in found.items() if v < minimum[k]}

    assert not short, (f"перевірки майже нічого не знайшли: {short} — схоже, "
                       "змінилась структура мапи, і вирази дивляться не туди")
