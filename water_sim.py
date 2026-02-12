import wntr
import numpy as np
import warnings
import os
import uuid
import sys
from ga_config import CONFIG

warnings.filterwarnings("ignore")

class WaterSimulator:
    def __init__(self, inp_file, temp_dir=None):
        self.inp_file = inp_file
        self.temp_dir = temp_dir 
        
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        self.graph = self.wn.get_graph()
        self.component_names = self.wn.pipe_name_list
        self.n_variables = len(self.component_names)
        
        self.lengths = np.array([self.wn.get_link(p).length for p in self.component_names])
        
        self.diams_m = []
        self.costs = []
        self.n_options = 0
        
        self.death_penalty = 1e16
        
        self._refresh_config()
        
        self.sources = self.wn.reservoir_name_list
        if not self.sources:
            self.sources = self.wn.tank_name_list

    def _refresh_config(self):
        if 'diameters_m' in CONFIG and len(CONFIG['diameters_m']) > 0:
            self.diams_m = CONFIG['diameters_m']
            if 'costs' in CONFIG and isinstance(CONFIG['costs'], dict):
                raw_diams = CONFIG.get('diameters_raw', [])
                self.costs = [CONFIG['costs'].get(d, 0.0) for d in raw_diams]
            else:
                self.costs = []
            self.n_options = len(self.diams_m)

    def _apply_pattern(self, individual):
        if not self.diams_m: self._refresh_config()
        
        for i, pipe_name in enumerate(self.component_names):
            idx = individual[i]
            if idx < 0: idx = 0
            if idx >= self.n_options: idx = self.n_options - 1
            self.wn.get_link(pipe_name).diameter = self.diams_m[idx]
            
        ind_arr = np.clip(individual, 0, self.n_options - 1)
        cost_arr = np.array(self.costs)
        total_cost = np.sum(self.lengths * cost_arr[ind_arr])
        return total_cost

    def _get_path(self, filename):
        if self.temp_dir: return os.path.join(self.temp_dir, filename)
        return filename

    def _clean_temp(self, prefix):
        for ext in [".inp", ".rpt", ".bin"]:
            try:
                fn = f"{prefix}{ext}"
                if os.path.exists(fn): os.remove(fn)
            except OSError: pass 

    def get_heuristics(self, individual):
        self._apply_pattern(individual)
        pid = os.getpid()
        prefix = self._get_path(f"heur_{pid}_{uuid.uuid4().hex[:8]}")
        
        sim = wntr.sim.EpanetSimulator(self.wn)
        try:
            results = sim.run_sim(file_prefix=prefix)
            headloss = results.link['headloss'].iloc[-1]
            unit_losses = []
            for name, length in zip(self.component_names, self.lengths):
                hl = abs(headloss[name])
                if length > 0: unit_losses.append(hl / length)
                else: unit_losses.append(0.0)
            return unit_losses
        except Exception:
            return [0.0] * self.n_variables
        finally:
            self._clean_temp(prefix)

    def evaluate(self, individual, penalty_factor=1000.0, epsilon=0.0, file_prefix=None):
        try: total_cost = self._apply_pattern(individual)
        except: return 1e16

        cleanup = False
        if file_prefix is None:
            pid = os.getpid()
            file_prefix = self._get_path(f"eval_{pid}_{uuid.uuid4().hex[:8]}")
            cleanup = True
        
        sim = wntr.sim.EpanetSimulator(self.wn)
        try:
            results = sim.run_sim(file_prefix=file_prefix)
            pressure = results.node['pressure'].iloc[-1]
            
            min_p = float('inf')
            max_violation = 0.0
            
            effective_limit = CONFIG.get('h_min', 30.0) - epsilon
            
            for node_name, node in self.wn.junctions():
                p = pressure[node_name]
                if p < min_p: min_p = p
                
                if p < effective_limit:
                    max_violation += (effective_limit - p)

            penalty = max_violation * penalty_factor
            return total_cost + penalty

        except Exception as e:
            return 1e15 
        finally:
            if cleanup: self._clean_temp(file_prefix)

    def get_stats(self, individual):
        cost = self._apply_pattern(individual)
        
        pid = os.getpid()
        prefix = self._get_path(f"stat_{pid}_{uuid.uuid4().hex[:8]}")
        
        sim = wntr.sim.EpanetSimulator(self.wn)
        try:
            results = sim.run_sim(file_prefix=prefix)
            pressure = results.node['pressure'].iloc[-1]
            
            min_p = float('inf')
            max_p = float('-inf')
            crit_node = None
            
            for node_name, node in self.wn.junctions():
                p = pressure[node_name]
                if p < min_p:
                    min_p = p
                    crit_node = node_name
                if p > max_p: max_p = p
            
            return cost, min_p, max_p, crit_node
        except:
            return cost, -1.0, -1.0, "ERR"
        finally:
            self._clean_temp(prefix)