import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, bw_range=(0.001, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_range = bw_range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = harmony_memory[np.random.randint(self.budget)][j]
                else:
                    if np.random.rand() < self.par:
                        new_harmony[j] = self.x_opt[j] + np.random.uniform(-self.bw, self.bw)
                    else:
                        new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
            self.bw = max(self.bw_range[0], self.bw * 0.99)  # Adaptive bandwidth update
        
        return self.f_opt, self.x_opt