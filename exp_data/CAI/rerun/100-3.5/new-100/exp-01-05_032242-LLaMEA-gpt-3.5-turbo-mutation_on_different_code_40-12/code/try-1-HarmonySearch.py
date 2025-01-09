import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, memory_acceptance=0.9):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.memory_acceptance = memory_acceptance
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        memory_indices = np.random.choice(self.dim, int(self.memory_acceptance * self.dim), replace=False)
        for i in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for d in range(self.dim):
                if d in memory_indices and np.random.rand() < self.hmcr:
                    new_harmony[d] = harmony_memory[np.random.choice(memory_indices)]
                    if np.random.rand() < self.par:
                        new_harmony[d] += self.bw * np.random.randn()
            f = func(new_harmony)
            if f < func(harmony_memory):
                harmony_memory = np.copy(new_harmony)
                memory_indices = np.random.choice(self.dim, int(self.memory_acceptance * self.dim), replace=False)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
        return self.f_opt, self.x_opt