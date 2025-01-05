import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, bw_range=(0.001, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.bw_range = bw_range  # Bandwidth Range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))

        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[d] = harmony_memory[d]
                else:
                    new_harmony[d] = np.random.uniform(func.bounds.lb, func.bounds.ub)

                if np.random.rand() < self.par:
                    new_harmony[d] += np.random.uniform(-self.bw, self.bw)

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony

                if np.random.rand() < 0.2:  # Adaptive bandwidth adjustment
                    self.bw = np.clip(self.bw + np.random.uniform(-0.005, 0.005), self.bw_range[0], self.bw_range[1])

        return self.f_opt, self.x_opt