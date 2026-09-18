"""`readme.md` не має описувати те, чого в коді немає.

Документація розійшлася з кодом на п'ять місяців і описувала модулі
`ga_*`, епохи, `make_warm_seeds`, `baseline_bonus` і функцію штрафу «для
GA-режиму» — усе прибране етапами 1, 9, 11 і 12. Для дисертації це
найдорожчий клас помилки: документ виглядає докладним, читач на нього
спирається, і дізнається про розбіжність у найгірший момент.

Тести нижче не перевіряють, що документ **повний** — це неможливо
автоматично. Вони перевіряють дві речі, які автоматизуються:

1. кожен модуль, згаданий у дереві проєкту, існує на диску;
2. жодне з прибраних імен не згадується як **робоче** (згадки у врізках
   «чого тут більше немає» дозволені навмисне — вони пояснюють читачеві,
   чому він не знаходить у коді того, що бачив у старій версії).
"""

import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "readme.md")

pytestmark = pytest.mark.skipif(not os.path.exists(README), reason="немає readme.md")


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def readme_paragraphs(readme):
    """Абзаци з номером першого рядка.

    Перевірка ведеться по абзацу, а не по рядку: пояснення «це прибрано
    на етапі N» майже завжди переноситься на наступний рядок, і
    порядкова перевірка давала б хибні падіння саме на правильно
    написаних врізках.
    """
    paragraphs = []
    line_no = 1
    for block in readme.split("\n\n"):
        paragraphs.append((line_no, block))
        line_no += block.count("\n") + 2
    return paragraphs


#: Імена, прибрані з коду. Ключ — ім'я, значення — де про це прочитати.
REMOVED = {
    "ga_config.py": "етап 11: перейменовано на config.py",
    "ga_data.py": "етап 11: перейменовано на cost_table.py",
    "ga_utils.py": "етап 11: перейменовано на console.py",
    "GAConfig": "етап 11: тепер SolverConfig",
    "make_warm_seeds": "етап 9: прибрано разом з епохами",
    "baseline_bonus": "етап 1: мертве поле",
    "greedy_path_relink": "етап 14: не викликалось ніде",
    "_build_diverse_archive": "етап 1: мертвий метод",
    "fast_crossover": "етап 14: не викликалось робочим кодом",
    "EPSILON_PRESSURE_WINDOW": "етап 15.09: механізм видалено після заміру",
}

#: Рядок вважається поясненням, а не описом робочого коду, якщо містить
#: одне з цих слів. Так врізки «чого більше немає» не дають хибних падінь.
EXPLANATORY = ("прибран", "видален", "перейменован", "більше немає", "мертв",
               "раніше", "доти", "до 15.09", "до версії", "розійш", "етап")


def is_explanatory(paragraph):
    lowered = paragraph.lower()
    return any(word in lowered for word in EXPLANATORY)


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_a_removed_name_is_only_mentioned_as_history(readme_paragraphs, name):
    offenders = [f"рядок {line_no}: {block.strip()[:90]}"
                 for line_no, block in readme_paragraphs
                 if name in block and not is_explanatory(block)]

    assert not offenders, (
        f"`{name}` ({REMOVED[name]}) згадується як робоче:\n  "
        + "\n  ".join(offenders)
        + "\n\nЯкщо це свідома згадка історії — поясніть у тому ж абзаці, "
          "коли і чому цього не стало.")


def test_every_module_in_the_project_tree_exists(readme):
    """Дерево проєкту в §1 — перше, що читають; воно мусить бути правдою."""
    tree = readme[readme.index("## 1. Структура проекту"):
                  readme.index("### Граф залежностей")]

    missing = []
    for match in re.finditer(r"([\w/]+\.py)\s+←", tree):
        name = match.group(1)
        # Ім'я в дереві коротке (`beam.py`), а лежати може в підпакеті —
        # тому шукається по всьому дереву проєкту, а не в двох місцях.
        candidates = [os.path.join(dirpath, name)
                      for dirpath, _, filenames in os.walk(ROOT)
                      if "__pycache__" not in dirpath
                      and os.path.basename(name) in filenames]
        candidates += [os.path.join(ROOT, name)]
        if not any(os.path.exists(c) for c in candidates):
            missing.append(name)

    assert not missing, f"у дереві є модулі, яких немає на диску: {missing}"


def test_every_solver_module_appears_in_the_tree(readme):
    """Зворотна перевірка: новий модуль не має з'явитись непоміченим.

    Обходяться і підпакети. Після розділення `island.py` на пакет
    перевірка, яка дивилась лише у верхній рівень, перестала б бачити
    п'ять модулів одразу — і мовчки зеленіла б далі.
    """
    tree = readme[readme.index("## 1. Структура проекту"):
                  readme.index("### Граф залежностей")]

    package = os.path.join(ROOT, "pipeopt", "solver")
    missing = []
    for dirpath, dirnames, filenames in os.walk(package):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            # `__init__.py` є в кожному пакеті; у дереві його досить один раз.
            if name == "__init__.py" and dirpath != package:
                continue
            if name not in tree:
                missing.append(os.path.relpath(os.path.join(dirpath, name), ROOT))

    assert not missing, (
        f"модулі є в коді, але не в дереві readme: {missing}")


def test_the_document_says_when_it_was_checked(readme):
    """Документ без дати звірки не дає читачеві оцінити, наскільки йому вірити."""
    assert re.search(r"\*\*Звірено з кодом:\*\*\s*\d{2}\.\d{2}\.\d{4}", readme)
