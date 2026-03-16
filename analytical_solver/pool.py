from .fast_math import fast_hamming_distance # 🔴 Додайте імпорт
import numpy as np

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
        """Створює унікальний підпис долини на основі макро-блоків (chunking)"""
        n = self.ctx.num_pipes
        if n <= 50:
            return tuple(indices) # Для малих мереж точний збіг
            
        # Для великих мереж розбиваємо на сектори і беремо середній діаметр
        chunk_size = max(1, n // 25)
        sig = []
        for i in range(0, n, chunk_size):
            chunk = indices[i:i+chunk_size]
            sig.append(round(sum(chunk) / len(chunk)))
            
        return tuple(sig)

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
        # Numba вимагає NumPy масиви
        arr1 = np.array(sol1, dtype=np.int32)
        arr2 = np.array(sol2, dtype=np.int32)
        return fast_hamming_distance(arr1, arr2)