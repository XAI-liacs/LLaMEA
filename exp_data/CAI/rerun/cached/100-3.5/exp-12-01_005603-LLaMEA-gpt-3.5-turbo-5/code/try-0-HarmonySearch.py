import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.4
        self.bandwidth = 0.01

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        harmony_fitness = np.apply_along_axis(func, 1, harmony_memory)

        for _ in range(self.budget):
            new_harmony = np.copy(harmony_memory[np.argmin(harmony_fitness)])
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[i] = new_harmony[i] + np.random.uniform(-self.bandwidth, self.bandwidth)

            new_fitness = func(new_harmony)
            if new_fitness < np.max(harmony_fitness):
                replace_idx = np.argmax(harmony_fitness)
                harmony_memory[replace_idx] = new_harmony
                harmony_fitness[replace_idx] = new_fitness

        return harmony_memory[np.argmin(harmony_fitness)]