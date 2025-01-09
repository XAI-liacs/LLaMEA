import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def improvise(self, func, harmony_memory):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                if np.random.rand() < self.par:
                    new_harmony[i] = harmony_memory[np.random.randint(len(harmony_memory))][i]
                else:
                    new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            else:
                new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                if np.random.rand() < self.bandwidth:
                    new_harmony[i] += np.random.normal(0, 1)

        return new_harmony

    def __call__(self, func):
        harmony_memory = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(10)]
        
        for _ in range(self.budget):
            new_harmony = self.improvise(func, harmony_memory)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory.append(new_harmony)
                harmony_memory = sorted(harmony_memory, key=lambda x: func(x))[:10]
            
        return self.f_opt, self.x_opt