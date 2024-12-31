import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.hmcr = 0.95
        self.par = 0.4
        self.bandwidth = 0.01

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        
        for i in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[j] = new_harmony[j] + np.random.uniform(-self.bandwidth, self.bandwidth)
                    else:
                        new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
             
            f = func(new_harmony)
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = np.copy(new_harmony)
                
        return self.f_opt, self.x_opt