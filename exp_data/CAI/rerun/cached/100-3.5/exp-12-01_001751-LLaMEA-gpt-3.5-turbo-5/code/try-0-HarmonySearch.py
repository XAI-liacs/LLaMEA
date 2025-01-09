import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bw
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony_memory(size):
            return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

        def update_harmony_memory(hm, new_solution):
            hm = np.vstack((hm, new_solution))
            hm_fitness = np.array([func(sol) for sol in hm])
            sorted_indices = np.argsort(hm_fitness)
            return hm[sorted_indices][:len(hm)]

        harmony_memory = initialize_harmony_memory(5)
        for _ in range(self.budget-5):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[i] = harmony_memory[np.random.randint(0, len(harmony_memory))][i]
                    if np.random.rand() < self.par:
                        new_solution[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                        new_solution[i] = np.clip(new_solution[i], self.lower_bound, self.upper_bound)
            harmony_memory = update_harmony_memory(harmony_memory, new_solution)
        best_solution = harmony_memory[np.argmin([func(sol) for sol in harmony_memory])]
        return best_solution