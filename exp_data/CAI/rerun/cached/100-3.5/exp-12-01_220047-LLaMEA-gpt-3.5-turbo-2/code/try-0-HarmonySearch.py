import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def generate_new_harmony(memory):
            new_harmony = np.copy(memory)
            for i in range(self.dim):
                if np.random.rand() < 0.7:  # pitch adjustment rate
                    new_harmony[np.random.randint(self.budget), i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory)
            evaluations = np.apply_along_axis(func, 1, new_harmony)
            best_index = np.argmin(evaluations)
            if best_solution is None or evaluations[best_index] < func(best_solution):
                best_solution = new_harmony[best_index]
            harmony_memory[np.argmax(evaluations)] = new_harmony[best_index]

        return best_solution