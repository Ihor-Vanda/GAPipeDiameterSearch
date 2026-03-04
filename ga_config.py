from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class GAConfig:
    inp_file: str = "InputData/Hanoi/Hanoi.inp"
    cost_file: str = "InputData/Hanoi/costs.csv"
    pop_size: int = 200
    n_gens: int = 150
    runs: int = 1
    h_min: float = 30.0
    
    mutation_start: float = 0.40
    mutation_end: float = 0.10
    epsilon_start: float = 10.0
    epsilon_end: float = 0.0
    
    run_mode: str = "ga"
    init_method: str = "sep"
    v_opt: float = 1.0
    analytical_mutation_rate: float = 0.15
    
    diameters_raw: List[float] = field(default_factory=list)
    diameters_m: List[float] = field(default_factory=list)
    costs: Dict[float, float] = field(default_factory=dict)
    unit_system: str = "mm"
    
    def get_max_diameter(self) -> float:
        return max(self.diameters_m) if self.diameters_m else 0.0

    def get_min_diameter(self) -> float:
        return min(self.diameters_m) if self.diameters_m else 0.0