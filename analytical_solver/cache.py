from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        
    def get(self, key):
        if key not in self.cache: return None
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def set(self, key, value):
        if key in self.cache: 
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)