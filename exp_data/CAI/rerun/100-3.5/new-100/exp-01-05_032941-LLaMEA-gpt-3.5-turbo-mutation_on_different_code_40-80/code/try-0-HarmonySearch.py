import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, hmcr=0.7, par=0.4, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, self.dim))
        for i in range(self.budget):
            if np.random.rand() < self.hmcr:
                idx = np.random.choice(self.harmony_memory_size)
                new_x = harmony_memory[idx]
                for j in range(self.dim):
                    if np.random.rand() < self.par:
                        new_x[j] = new_x[j] + np.random.uniform(-self.bandwidth, self.bandwidth)
                        new_x[j] = np.clip(new_x[j], func.bounds.lb[j], func.bounds.ub[j])
            else:
                new_x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_x
                harmony_memory[np.argmax(harmony_memory)] = new_x
                
        return self.f_opt, self.x_opt