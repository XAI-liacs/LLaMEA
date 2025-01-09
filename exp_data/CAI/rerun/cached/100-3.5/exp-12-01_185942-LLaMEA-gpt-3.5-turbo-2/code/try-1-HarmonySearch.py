import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def improvise_new_value():
            new_value = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            return new_value

        harmony_memory = [improvise_new_value() for _ in range(self.budget)]
        harmony_memory_fitness = [func(harmony) for harmony in harmony_memory]

        while self.budget > 0:
            new_harmony = improvise_new_value()
            new_harmony_fitness = func(new_harmony)

            if new_harmony_fitness < max(harmony_memory_fitness):
                max_index = np.argmax(harmony_memory_fitness)
                harmony_memory[max_index] = new_harmony
                harmony_memory_fitness[max_index] = new_harmony_fitness

            self.budget -= 1

        best_index = np.argmin(harmony_memory_fitness)
        best_solution = harmony_memory[best_index]
        return best_solution