class AbstractSimulator:
    def __init__(self):
        self.n_variables = 0     
        self.n_options = 0      
        self.graph = None        
        self.sources = []        
        self.component_names = [] 

    def evaluate(self, individual, penalty_factor=1.0):
        raise NotImplementedError

    def get_stats(self, individual):
        raise NotImplementedError

    def get_heuristics(self, individual):
        raise NotImplementedError