"""Синтетична мережа-сітка, більша за Балерму.

Потрібна, щоб перевіряти масштабованість порогів: Балерма (454 труби) —
найбільше, що є в наборі, а майбутні мережі можуть бути більшими. Сітка
взята навмисно «важкою» топологічно: у неї багато контурів (для 24×24
близько 530), тоді як у Балерми їх 8, а в Ханої 3. Якщо порогові правила
залежать від розміру мережі, саме тут це має бути видно.

Генерується у одиницях Балерми (мм, h_min = 20 м), тією самою таблицею
цін, тому мережі порівнянні.

    python experiments/make_grid_network.py --rows 24 --cols 24
"""

import argparse
import os


def build(rows, cols, spacing=100.0, demand_lps=0.35, source_head=95.0,
          ground_elevation=0.0, roughness=130.0, diameter_mm=200.0):
    """Зібрати .inp сітки rows×cols із одним резервуаром у куті.

    Труби отримують однаковий стартовий діаметр — оптимізатор його все
    одно перезаписує; важливо лише, щоб файл був коректним EPANET.
    """
    junctions = []
    for r in range(rows):
        for c in range(cols):
            junctions.append((f"J{r}_{c}", r, c))

    pipes = []
    index = 0
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                index += 1
                pipes.append((f"P{index}", f"J{r}_{c}", f"J{r}_{c + 1}"))
            if r + 1 < rows:
                index += 1
                pipes.append((f"P{index}", f"J{r}_{c}", f"J{r + 1}_{c}"))

    # Резервуар живить кут сітки коротким магістральним відводом.
    index += 1
    source_pipe = (f"P{index}", "R1", "J0_0")

    lines = ["[TITLE]", f"Synthetic grid {rows}x{cols}", "", "[JUNCTIONS]",
             ";ID              Elev        Demand      Pattern"]
    for name, r, c in junctions:
        # Легкий нахил ділянки, щоб задача не була симетричною.
        elevation = ground_elevation + 0.02 * (r + c) * spacing / 10.0
        lines.append(f" {name:<15} {elevation:<11.3f} {demand_lps:<11.3f} ;")

    lines += ["", "[RESERVOIRS]", ";ID              Head",
              f" R1              {source_head:.3f} ;", "", "[TANKS]", ";ID",
              "", "[PIPES]",
              ";ID              Node1           Node2           Length      Diameter    Roughness   MinorLoss   Status"]
    for name, a, b in pipes:
        lines.append(f" {name:<15} {a:<15} {b:<15} {spacing:<11.3f} "
                     f"{diameter_mm:<11.3f} {roughness:<11.3f} 0           Open ;")
    name, a, b = source_pipe
    lines.append(f" {name:<15} {a:<15} {b:<15} {spacing / 2:<11.3f} "
                 f"{600.0:<11.3f} {roughness:<11.3f} 0           Open ;")

    lines += ["", "[PUMPS]", ";ID", "", "[VALVES]", ";ID", "", "[PATTERNS]", ";ID",
              "", "[CURVES]", ";ID", "", "[CONTROLS]", "", "[RULES]", "",
              "[ENERGY]", " Global Efficiency  75", "", "[EMITTERS]", ";ID",
              "", "[QUALITY]", ";Node", "", "[SOURCES]", ";Node", "",
              "[REACTIONS]", " Order Bulk            1", "", "[MIXING]", ";Tank",
              "", "[TIMES]", " Duration           0", " Hydraulic Timestep 1:00",
              " Pattern Timestep   1:00", " Report Timestep    1:00",
              " Start ClockTime    12 am", "", "[REPORT]", " Status             No",
              " Summary            No", " Page               0", "",
              "[OPTIONS]", " Units              LPS", " Headloss           H-W",
              " Specific Gravity   1.0", " Viscosity          1.0",
              " Trials             200", " Accuracy           0.001",
              " Unbalanced         Continue 10", " Pattern            1",
              " Demand Multiplier  1.0", " Emitter Exponent   0.5",
              " Quality            None mg/L", " Diffusivity        1.0",
              " Tolerance          0.01", "", "[COORDINATES]", ";Node X Y"]
    for name, r, c in junctions:
        lines.append(f" {name:<15} {c * spacing:<11.2f} {r * spacing:<11.2f}")
    lines.append(f" R1              {-spacing:<11.2f} {-spacing:<11.2f}")
    lines += ["", "[VERTICES]", ";Link", "", "[LABELS]", "", "[BACKDROP]", "",
              "[TAGS]", "", "[END]", ""]

    return "\n".join(lines), len(pipes) + 1, len(junctions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--cols", type=int, default=24)
    parser.add_argument("--demand", type=float, default=0.35,
                        help="Відбір у кожному вузлі, л/с")
    parser.add_argument("--head", type=float, default=95.0,
                        help="Напір у резервуарі, м")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    text, n_pipes, n_junctions = build(args.rows, args.cols,
                                       demand_lps=args.demand, source_head=args.head)
    name = f"Grid{args.rows}x{args.cols}"
    out_dir = args.out or os.path.join("InputData", name)
    os.makedirs(out_dir, exist_ok=True)

    inp_path = os.path.join(out_dir, f"{name}.inp")
    with open(inp_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    # Таблиця цін — та сама, що в Балерми, щоб мережі були порівнянні.
    costs = """Diameter,Cost
113.0, 7.22
126.6, 9.10
144.6, 11.92
162.8, 14.84
180.8, 18.38
226.2, 28.60
285.0, 45.39
361.8, 76.32
452.2, 124.64
581.8, 215.85
"""
    with open(os.path.join(out_dir, "costs.csv"), "w", encoding="utf-8") as handle:
        handle.write(costs)

    print(f"{inp_path}: {n_pipes} труб, {n_junctions} вузлів")


if __name__ == "__main__":
    main()
