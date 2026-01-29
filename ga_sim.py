import wntr
import warnings
from ga_config import CONFIG

class NetworkSimulator:
    def __init__(self, inp_file):
        self.inp_file = inp_file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(self.inp_file)
            self.pipe_names = wn.pipe_name_list
            self.junction_names = wn.junction_name_list
            self.n_pipes = len(self.pipe_names)

    def evaluate(self, individual, strategy="static", gen=0, tolerance=0.0, total_gens=100):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(self.inp_file)
            
            total_cost = 0.0
            diams_raw = CONFIG["diameters_raw"]
            diams_m = CONFIG["diameters_m"]
            costs = CONFIG["costs"]
            h_min = CONFIG["h_min"]

            for i, pipe_name in enumerate(self.pipe_names):
                idx = individual[i]
                total_cost += wn.get_link(pipe_name).length * costs[diams_raw[idx]]
                wn.get_link(pipe_name).diameter = diams_m[idx]

            sim = wntr.sim.EpanetSimulator(wn)
            try:
                results = sim.run_sim()
            except Exception:
                return 1e15, 
            
            pressures = results.node['pressure'].iloc[-1]
            violation = 0.0
            effective_limit = h_min - tolerance
            
            for node in self.junction_names:
                p = pressures[node]
                if p < effective_limit:
                    violation += (effective_limit - p)

            penalty = 0.0
            if violation > 0:
                penalty = 1e9 + (1e6 * violation)

            return total_cost + penalty,

    def get_stats(self, individual):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wn = wntr.network.WaterNetworkModel(self.inp_file)
            
            total_cost = 0.0
            diams_raw = CONFIG["diameters_raw"]
            diams_m = CONFIG["diameters_m"]
            costs = CONFIG["costs"]
            
            for i, pipe_name in enumerate(self.pipe_names):
                idx = individual[i]
                total_cost += wn.get_link(pipe_name).length * costs[diams_raw[idx]]
                wn.get_link(pipe_name).diameter = diams_m[idx]
            
            try:
                sim = wntr.sim.EpanetSimulator(wn)
                results = sim.run_sim()
                min_p = results.node['pressure'].iloc[-1].loc[self.junction_names].min()
            except:
                min_p = -999.0
            
            return total_cost, min_p