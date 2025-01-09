import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, memory_size=10):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.memory_size = memory_size
        self.f_opt = np.Inf
        self.x_opt = None

    def improvisation(self, func, harmony_memory):
        new_harmony = np.zeros(self.dim)
        for d in range(self.dim):
            if np.random.rand() < self.hmcr:
                new_harmony[d] = harmony_memory[np.random.randint(len(harmony_memory))][d]
                if np.random.rand() < self.par:
                    new_harmony[d] = new_harmony[d] + np.random.uniform(-self.bw, self.bw)
            else:
                new_harmony[d] = np.random.uniform(func.bounds.lb, func.bounds.ub)
        return new_harmony

    def update_memory(self, harmony_memory, new_harmony, func):
        harmony_memory.append(new_harmony)
        harmony_memory.sort(key=lambda x: func(x))
        harmony_memory = harmony_memory[:self.memory_size]
        return harmony_memory

    def __call__(self, func):
        harmony_memory = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.memory_size)]
        
        for i in range(self.budget):
            new_harmony = self.improvisation(func, harmony_memory)
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = self.update_memory(harmony_memory, new_harmony, func)
        
        return self.f_opt, self.x_opt