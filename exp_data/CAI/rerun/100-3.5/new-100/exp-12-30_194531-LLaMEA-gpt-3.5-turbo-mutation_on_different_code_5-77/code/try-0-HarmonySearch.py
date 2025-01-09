import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.5, bw=0.02):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  
        self.par = par  
        self.bw = bw  
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            if np.random.rand() < self.hmcr:
                new_harmony = np.random.uniform(func.bounds.lb, func.bounds.ub) if np.random.rand() < self.par else self.x_opt + np.random.uniform(-self.bw, self.bw)
            else:
                new_harmony = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
        return self.f_opt, self.x_opt