import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.05):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        
        for _ in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[np.random.randint(self.dim)]
                    if np.random.rand() < self.par:
                        new_harmony[i] += self.bw * np.random.randn()
                        
            f = func(new_harmony)
            if f < func(harmony_memory):
                harmony_memory = new_harmony
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                
            if np.random.rand() < 0.1:  # Adaptive bandwidth adjustment
                self.bw *= 0.9  # Reduce bandwidth slightly
            
        return self.f_opt, self.x_opt