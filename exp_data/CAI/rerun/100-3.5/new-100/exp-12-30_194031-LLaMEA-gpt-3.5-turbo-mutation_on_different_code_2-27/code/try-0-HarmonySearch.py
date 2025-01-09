import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.2, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        harmony_memory_fitness = np.array([func(x) for x in harmony_memory])

        for _ in range(self.budget):
            new_harmony = np.copy(harmony_memory[np.random.randint(len(harmony_memory))])
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[i] = new_harmony[i] + np.random.uniform(-self.bw, self.bw)
                    else:
                        new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            new_fitness = func(new_harmony)
            idx = np.argmin(harmony_memory_fitness)
            if new_fitness < harmony_memory_fitness[idx]:
                harmony_memory[idx] = new_harmony
                harmony_memory_fitness[idx] = new_fitness

        idx = np.argmin(harmony_memory_fitness)
        self.x_opt = harmony_memory[idx]
        self.f_opt = harmony_memory_fitness[idx]

        return self.f_opt, self.x_opt