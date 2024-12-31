import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.01, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))

        for i in range(self.budget):
            if np.random.rand() < self.hmcr:
                idx = np.random.choice(self.budget)
                if np.random.rand() < self.par:
                    harmony_memory[i] = harmony_memory[idx] + np.random.uniform(-self.bw, self.bw, self.dim)
                else:
                    harmony_memory[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                
            f = func(harmony_memory[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = harmony_memory[i]
            
        return self.f_opt, self.x_opt