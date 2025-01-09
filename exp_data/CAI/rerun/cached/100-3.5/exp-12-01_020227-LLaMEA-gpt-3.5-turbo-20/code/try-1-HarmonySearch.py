import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def generate_harmony(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
    
    def __call__(self, func):
        harmonies = [self.generate_harmony() for _ in range(self.budget)]
        best_harmony = min(harmonies, key=func)
        return best_harmony