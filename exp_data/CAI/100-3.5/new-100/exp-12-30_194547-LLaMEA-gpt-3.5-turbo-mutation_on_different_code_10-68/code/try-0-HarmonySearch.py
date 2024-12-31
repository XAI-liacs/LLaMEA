import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, hmcr=0.95, par=0.5, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim))
        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[j] = harmony_memory[idx, j]
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)

                if np.random.rand() < self.par:
                    new_harmony[j] += np.random.uniform(-self.bandwidth, self.bandwidth)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory[np.argmax(harmony_memory[:, -1])] = new_harmony

        return self.f_opt, self.x_opt