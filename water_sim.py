import wntr
import numpy as np
import warnings
import os

from wntr.epanet.toolkit import ENepanet

from sim_stats import SimStats

warnings.filterwarnings("ignore")


EN_NODECOUNT = 0
EN_LINKCOUNT = 2
EN_JUNCTION = 0
EN_DIAMETER = 0
EN_LENGTH = 1
EN_FLOW = 8
EN_HEADLOSS = 10
EN_PRESSURE = 11

class WaterSimulator:
    def __init__(self, inp_file, config, temp_dir):
        """`temp_dir` обов'язковий: з `None` робочі файли EPANET писались у
        поточний каталог, тобто в корінь проєкту. Усі шість місць виклику й
        так його передають, тому замовчування було лише пасткою (§29.2)."""
        self.inp_file = inp_file
        self.config = config
        self.temp_dir = temp_dir
        
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        # `to_graph`, не `get_graph`: друге застаріле і друкує
        # DeprecationWarning на кожен розбір мережі. Перевірено, що вузли,
        # ребра й тип графа тотожні (§30.5).
        self.graph = self.wn.to_graph()
        self.component_names = self.wn.pipe_name_list
        self.n_variables = len(self.component_names)
        
        self.lengths = np.array([self.wn.get_link(p).length for p in self.component_names])
        
        rpt_file = os.path.join(temp_dir, f"temp_rpt_{os.getpid()}.rpt")
        bin_file = os.path.join(temp_dir, f"temp_bin_{os.getpid()}.bin")
        
        self.api = ENepanet(self.inp_file, rpt_file, bin_file)
        self.api.ENopen()
        
        self.n_nodes = self.api.ENgetcount(EN_NODECOUNT)
        self.n_links = self.api.ENgetcount(EN_LINKCOUNT)
        
        flow_units = self.api.ENgetflowunits()
        self.is_us_units = flow_units < 5
        
        if self.is_us_units:
            self.diam_epanet_mult = 39.3701 
            self.press_si_mult = 0.7032496    
            self.hl_si_mult = 0.3048        
        else:
            self.diam_epanet_mult = 1000.0   
            self.press_si_mult = 1.0         
            self.hl_si_mult = 1.0         

        self.pipe_name_to_c_idx = {}
        for name in self.component_names:
            c_idx = self.api.ENgetlinkindex(name)
            self.pipe_name_to_c_idx[name] = c_idx
            
        self.junction_c_indices = []
        self.c_idx_to_node_name = {}
        
        self.sources = self.wn.reservoir_name_list
        if not self.sources:
            self.sources = self.wn.tank_name_list
            
        for name in self.wn.junction_name_list:
            c_idx = self.api.ENgetnodeindex(name)
            self.junction_c_indices.append(c_idx)
            self.c_idx_to_node_name[c_idx] = name
                    
        self.diams_m = []
        self.costs = []
        self.n_options = 0

        self._refresh_config()

    def _refresh_config(self):
        if self.config:
            if len(self.config.diameters_m) > 0:
                self.diams_m = self.config.diameters_m
            if isinstance(self.config.costs, dict):
                raw_diams = self.config.diameters_raw
                self.costs = [self.config.costs.get(d, 0.0) for d in raw_diams]
            else:
                self.costs = []
            self.n_options = len(self.diams_m)
            # Масив цін використовується в кожній симуляції; раніше він
            # створювався з списку **на кожен виклик** `_apply_pattern`,
            # тобто мільйон разів за прогін (§30.3).
            self.cost_array = np.array(self.costs, dtype=np.float64)

    def __del__(self):
        try:
            if hasattr(self, 'api') and self.api is not None:
                self.api.ENclose()
        except:
            pass

    def _apply_pattern(self, individual):
        if not self.diams_m: self._refresh_config()
        
        ind_arr = np.clip(individual, 0, self.n_options - 1)
        total_cost = np.sum(self.lengths * self.cost_array[ind_arr])
        
        for i, pipe_name in enumerate(self.component_names):
            c_idx = self.pipe_name_to_c_idx[pipe_name]
            idx = ind_arr[i]
            val = self.diams_m[idx] * self.diam_epanet_mult
            self.api.ENsetlinkvalue(c_idx, EN_DIAMETER, val)
            
        return total_cost

    def _run_simulation_core(self):
        try:
            self.api.ENopenH()
            self.api.ENinitH(0)
            
            while True:
                self.api.ENrunH()
                tstep = self.api.ENnextH()
                if tstep <= 0:
                    break
                    
            self.api.ENcloseH()
            return True
        except Exception:
            return False

    def get_heuristics(self, individual):
        self._apply_pattern(individual)
        ok = self._run_simulation_core()
        
        if not ok: 
            return [0.0] * self.n_variables
            
        unit_losses = []
        for i, pipe_name in enumerate(self.component_names):
            c_idx = self.pipe_name_to_c_idx[pipe_name]
            hl = abs(self.api.ENgetlinkvalue(c_idx, EN_HEADLOSS)) * self.hl_si_mult
            L = self.lengths[i]
            unit_losses.append(hl / L if L > 0 else 0.0)
            
        return unit_losses

    def get_stats(self, individual):
        cost = self._apply_pattern(individual)
        ok = self._run_simulation_core()

        if not ok:
            return SimStats(cost, -1.0, -1.0, "ERR")

        min_p = float('inf')
        max_p = float('-inf')
        crit_node_idx = -1
        
        for c_idx in self.junction_c_indices:
            p = self.api.ENgetnodevalue(c_idx, EN_PRESSURE) * self.press_si_mult
            if p < min_p:
                min_p = p
                crit_node_idx = c_idx
            if p > max_p: 
                max_p = p
                
        crit_node = self.c_idx_to_node_name.get(crit_node_idx, "ERR")

        return SimStats(cost, min_p, max_p, crit_node)

    def get_hydraulic_state(self, individual_diameters):
        for i, pipe_name in enumerate(self.component_names):
            c_idx = self.pipe_name_to_c_idx[pipe_name]
            val = individual_diameters[i] * self.diam_epanet_mult
            self.api.ENsetlinkvalue(c_idx, EN_DIAMETER, val)
            
        ok = self._run_simulation_core()
        if not ok: return [0.0] * self.n_variables, False
            
        flows = []
        for i, pipe_name in enumerate(self.component_names):
            c_idx = self.pipe_name_to_c_idx[pipe_name]
            flows.append(self.api.ENgetlinkvalue(c_idx, EN_FLOW))
            
        min_p = float('inf')
        for c_idx in self.junction_c_indices:
            p = self.api.ENgetnodevalue(c_idx, EN_PRESSURE) * self.press_si_mult
            if p < min_p: min_p = p
            
        is_feasible = min_p >= self.config.h_min
        
        return flows, is_feasible