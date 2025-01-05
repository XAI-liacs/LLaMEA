import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.1):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            if np.random.rand() < self.hmcr:
                x = harmony_memory[np.random.choice(self.budget)]
            else:
                rand_idx = np.random.choice(self.budget, size=2, replace=False)
                x = np.clip(harmony_memory[rand_idx[0]] + np.random.uniform(-self.par, self.par, self.dim),
                            func.bounds.lb, func.bounds.ub)

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt