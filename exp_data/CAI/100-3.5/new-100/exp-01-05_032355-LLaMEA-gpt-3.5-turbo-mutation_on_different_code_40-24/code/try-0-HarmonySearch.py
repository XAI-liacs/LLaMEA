import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        
        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[d] = harmony_memory[d]
                else:
                    rand_idx = np.random.randint(0, len(harmony_memory))
                    new_harmony[d] = harmony_memory[rand_idx] + np.random.uniform(-self.bw, self.bw)
                    new_harmony[d] = np.clip(new_harmony[d], func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony
                
        return self.f_opt, self.x_opt