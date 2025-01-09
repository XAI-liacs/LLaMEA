import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.5, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        
        for i in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        index = np.random.randint(self.dim)
                        new_harmony[j] = harmony_memory[index] + np.random.uniform(-self.bw, self.bw)
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony
            self.bw *= 0.995  # Adaptive bandwidth adjustment for better convergence
        
        return self.f_opt, self.x_opt