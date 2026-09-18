from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SolverConfig:
    """Конфігурація задачі: обмеження, одиниці та таблиця діаметрів/цін.

    Заповнюється частково з CLI/GUI, частково у `load_cost_table()` (поля
    `diameters_raw`, `diameters_m`, `costs`).
    """

    cost_file: str = "InputData/Hanoi/costs.csv"
    h_min: float = 30.0
    run_mode: str = "analytical"
    v_opt: float = 1.0
    unit_system: str = "mm"

    diameters_raw: List[float] = field(default_factory=list)
    diameters_m: List[float] = field(default_factory=list)
    costs: Dict[float, float] = field(default_factory=dict)
