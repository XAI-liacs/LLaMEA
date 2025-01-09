import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)

        for i in range(self.budget):
            new_harmony = np.zeros_like(harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = harmony_memory[j]
                else:
                    rand_idx = np.random.randint(0, self.dim)
                    new_harmony[j] = harmony_memory[rand_idx]
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-self.bw, self.bw)

            new_harmony_fitness = func(new_harmony)

            if new_harmony_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_harmony_fitness

            if harmony_memory_fitness < self.f_opt:
                self.f_opt = harmony_memory_fitness
                self.x_opt = harmony_memory.copy()

        return self.f_opt, self.x_opt