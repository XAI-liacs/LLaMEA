import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hms=10, hmcr=0.7, par=0.01):
        self.budget = budget
        self.dim = dim
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmonies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.hms, self.dim))
        
        for _ in range(self.budget):
            new_harmony = np.zeros((1, self.dim))
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[0, j] = harmonies[np.random.randint(self.hms), j]
                else:
                    new_harmony[0, j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    if np.random.rand() < self.par:
                        new_harmony[0, j] = new_harmony[0, j] + np.random.normal(0, 1)
                    new_harmony[0, j] = np.clip(new_harmony[0, j], func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony.squeeze())
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony.squeeze()
            
        return self.f_opt, self.x_opt