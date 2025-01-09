import numpy as np

class ImprovedHarmonySearchAMEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.bandwidth = 0.01
        self.par_min = 0.3
        self.par_max = 0.9
        self.hmCR = 0.95
        self.evap_rate = 0.7

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def explore_new_solution(harmony_memory):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                r = np.random.rand()
                if r < self.hmCR:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_solution[d] = harmony_memory[idx, d]
                else:
                    new_solution[d] = np.random.uniform(-5.0, 5.0)
            return new_solution

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        for _ in range(self.budget):
            new_harmony = explore_new_solution(harmony_memory)
            new_fitness = objective_function(new_harmony)
            if best_solution is None or new_fitness < objective_function(best_solution):
                best_solution = new_harmony.copy()
            worst_idx = np.argmax([objective_function(h) for h in harmony_memory])
            if new_fitness < objective_function(harmony_memory[worst_idx]):
                harmony_memory[worst_idx] = new_harmony
            adapt_rate = 1 - (objective_function(new_harmony) - objective_function(best_solution)) / self.budget
            harmony_memory = harmony_memory * adapt_rate + best_solution * (1 - adapt_rate)
        return best_solution