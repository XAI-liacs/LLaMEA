import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        
        for _ in range(self.budget):
            new_harmony = np.where(np.random.rand(self.dim) < self.hmcr, harmony_memory, np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,)))
            for i in range(self.dim):
                if np.random.rand() < self.par:
                    new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony
            
        return self.f_opt, self.x_opt