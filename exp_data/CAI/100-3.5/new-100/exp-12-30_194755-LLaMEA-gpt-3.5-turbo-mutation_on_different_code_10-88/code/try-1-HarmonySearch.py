import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, bw_range=(0.01, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_range = bw_range  # Adaptive bandwidth control range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        for i in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    rand_index = np.random.randint(0, len(harmony_memory))
                    new_harmony[j] = harmony_memory[rand_index]
                    if np.random.rand() < self.par:
                        bw_range = np.abs(func.bounds.ub - func.bounds.lb) * self.bw_range
                        new_harmony[j] += np.random.uniform(-bw_range, bw_range)
            
            f = func(new_harmony)
            if f < func(harmony_memory):
                harmony_memory = np.copy(new_harmony)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_harmony
            
        return self.f_opt, self.x_opt