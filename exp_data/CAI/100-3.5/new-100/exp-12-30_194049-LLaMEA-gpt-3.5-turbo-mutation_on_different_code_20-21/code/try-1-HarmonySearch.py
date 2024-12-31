import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, bw_range=(0.01, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.bw_range = bw_range  # Bandwidth Range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        
        for i in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    idx = np.random.choice(self.dim)
                    new_harmony[j] = harmony_memory[idx]
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-self.bw, self.bw)
                        self.bw = max(self.bw_range[0], min(self.bw_range[1], self.bw + np.random.normal(0, 1)))
            
            f = func(new_harmony)
            if f < func(harmony_memory):
                harmony_memory = new_harmony
                
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
        return self.f_opt, self.x_opt