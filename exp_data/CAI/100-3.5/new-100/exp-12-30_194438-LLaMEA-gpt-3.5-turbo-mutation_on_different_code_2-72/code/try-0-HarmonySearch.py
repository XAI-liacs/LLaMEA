import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par    # Pitch Adjustment Rate
        self.bandwidth = bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        
        for _ in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[np.random.randint(self.dim)]
                if np.random.rand() < self.par:
                    new_harmony[i] += self.bandwidth * np.random.randn()
                    new_harmony[i] = np.clip(new_harmony[i], func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < func(harmony_memory):
                harmony_memory = np.copy(new_harmony)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_harmony
            
        return self.f_opt, self.x_opt