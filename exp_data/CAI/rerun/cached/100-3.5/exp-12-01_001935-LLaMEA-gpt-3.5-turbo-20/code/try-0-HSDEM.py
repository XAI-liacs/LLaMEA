import numpy as np

class HSDEM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_harmony_memory(hms_size):
            return np.random.uniform(self.lower_bound, self.upper_bound, (hms_size, self.dim))

        def update_harmony_memory(hms, new_solution):
            sorted_indices = np.argsort([objective_function(sol) for sol in hms])
            hms[sorted_indices[0]] = new_solution

        harmony_memory_size = 10
        harmony_memory = initialize_harmony_memory(harmony_memory_size)

        for _ in range(self.budget):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            update_harmony_memory(harmony_memory, new_solution)

        best_solution = min(harmony_memory, key=lambda x: objective_function(x))
        return best_solution