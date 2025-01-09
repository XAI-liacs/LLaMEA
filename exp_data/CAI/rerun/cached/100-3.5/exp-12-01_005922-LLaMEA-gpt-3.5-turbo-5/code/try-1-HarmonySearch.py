import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.3, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def update_harmony_memory(harmony_memory, new_solution):
            idx = np.argmax(func(harmony_memory))
            if func(new_solution) < func(harmony_memory[idx]):
                harmony_memory[idx] = new_solution
            return harmony_memory

        harmony_memory = initialize_harmony_memory()

        for _ in range(self.budget):
            new_solution = np.where(np.random.rand(self.dim) < self.hmcr, np.random.uniform(-5.0, 5.0, self.dim),
                                    harmony_memory[np.random.randint(self.budget)])

            mask = np.random.rand(self.dim) < self.par
            new_solution = new_solution + (mask * np.random.uniform(-1.0, 1.0) * self.bw)

            harmony_memory = update_harmony_memory(harmony_memory, new_solution)

        best_solution = harmony_memory[np.argmin(func(harmony_memory))]
        return best_solution