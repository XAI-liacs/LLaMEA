import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, bw_range=(0.01, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_range = bw_range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        
        for i in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[d] = harmony_memory[d]
                else:
                    if np.random.rand() < self.par:
                        idx = np.random.randint(0, len(harmony_memory))
                        new_harmony[d] = harmony_memory[idx]
                    else:
                        new_harmony[d] = np.random.uniform(func.bounds.lb[d], func.bounds.ub[d])
                        new_harmony[d] += np.random.uniform(-self.bw, self.bw)
                        new_harmony[d] = np.clip(new_harmony[d], func.bounds.lb[d], func.bounds.ub[d])

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony

            # Adaptive bandwidth adjustment
            if i % 100 == 0:
                self.bw = max(self.bw_range[0], self.bw - (self.bw_range[1] - self.bw_range[0]) * i / self.budget)

        return self.f_opt, self.x_opt