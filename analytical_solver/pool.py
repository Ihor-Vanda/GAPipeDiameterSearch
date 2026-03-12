class SolutionPool:
    def __init__(self, ctx):
        self.ctx = ctx
        self.active_pool = []
        self.tabu_fingerprints = {} # 🔴 Тепер це словник {fingerprint: round_added}
        self.kick_tabu_set = set()
        self.basin_tabu = set() 
        self.current_round = 0
        
    def clear_all(self):
        self.active_pool.clear()
        self.tabu_fingerprints.clear()
        self.kick_tabu_set.clear()
        self.basin_tabu.clear()

    def get_basin_signature(self, indices):
        """Створює унікальний підпис долини на основі гідравлічного скелета"""
        n = self.ctx.num_pipes
        top_k = max(3, n // 30)
        
        # Визначаємо структурно найважливіші труби для цього рішення
        unit_losses = self.ctx.get_cached_heuristics(indices)
        top_pipes = sorted(range(n), key=lambda i: unit_losses[i], reverse=True)[:top_k]
        
        return tuple(indices[p] for p in sorted(top_pipes))

    def is_basin_tabu(self, indices):
        return self.get_basin_signature(indices) in self.basin_tabu

    def add_basin_to_tabu(self, indices):
        self.basin_tabu.add(self.get_basin_signature(indices))

    def add_to_tabu(self, indices, cost):
        fp = self.ctx.get_fingerprint(indices, cost)
        self.tabu_fingerprints[fp] = self.current_round

    def is_tabu(self, indices, cost, tenure=80):
        fp = self.ctx.get_fingerprint(indices, cost)
        added_at = self.tabu_fingerprints.get(fp)
        if added_at is None: 
            return False
        # 🔴 Рішення забувається через 80 раундів
        return (self.current_round - added_at) < tenure
    
    def hamming_distance(self, sol1, sol2):
        return sum(1 for a, b in zip(sol1, sol2) if a != b)