import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.bandwidth = 0.01
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.7:
                    new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                else:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_solution[i] = self.harmony_memory[idx, i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                    new_solution[i] = max(min(new_solution[i], self.upper_bound), self.lower_bound)
            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[np.argsort([func(x) for x in self.harmony_memory])]
        return self.harmony_memory[0]