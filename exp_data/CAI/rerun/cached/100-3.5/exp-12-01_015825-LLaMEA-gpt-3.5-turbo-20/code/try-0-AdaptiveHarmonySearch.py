import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.bandwidth = 0.01
        self.pitch_adjustment_rate = 0.5

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def generate_new_harmony(harmony_memory):
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[:, i] += np.random.uniform(-self.bandwidth, self.bandwidth, self.harmony_memory_size)
                    new_harmony[:, i] = np.clip(new_harmony[:, i], -5.0, 5.0)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf
        num_evaluations = 0

        while num_evaluations < self.budget:
            new_harmony = generate_new_harmony(harmony_memory)
            new_fitness = np.array([func(h) for h in new_harmony])
            num_evaluations += self.harmony_memory_size

            if np.min(new_fitness) < best_fitness:
                best_fitness = np.min(new_fitness)
                best_solution = new_harmony[np.argmin(new_fitness)]

            harmony_memory = np.vstack((harmony_memory, new_harmony))
            harmony_memory = harmony_memory[np.argsort(new_fitness)[:self.harmony_memory_size]]

        return best_solution