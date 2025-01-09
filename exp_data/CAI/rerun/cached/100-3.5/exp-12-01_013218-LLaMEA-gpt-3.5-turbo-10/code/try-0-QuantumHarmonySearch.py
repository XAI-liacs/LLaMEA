import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = np.random.uniform(-5, 5, (budget, dim))
        self.harmony_memory_consideration_rate = 0.95
        self.bandwidth = 0.01
        
    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.uniform() < self.harmony_memory_consideration_rate:
                    new_solution[d] = self.harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.random.uniform(-5, 5)
                if np.random.uniform() < self.bandwidth:
                    new_solution[d] += np.random.normal(0, 1)
            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[np.argsort([func(sol) for sol in self.harmony_memory])]
        return self.harmony_memory[0]