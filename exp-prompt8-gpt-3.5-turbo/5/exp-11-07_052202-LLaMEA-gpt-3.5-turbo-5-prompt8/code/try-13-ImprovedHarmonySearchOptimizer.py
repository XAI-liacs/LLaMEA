import numpy as np

class ImprovedHarmonySearchOptimizer:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_bandwidth = pitch_bandwidth

    def __call__(self, func):
        self.harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        self.fitness_memory = np.array([func(harmony) for harmony in self.harmony_memory])

        for _ in range(self.budget - self.harmony_memory_size):
            indexes = np.random.rand(self.dim) < self.pitch_adjust_rate
            selected_indexes = np.random.randint(self.harmony_memory_size, size=self.dim)
            new_harmony = np.where(indexes, self.harmony_memory[selected_indexes], np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth, self.dim))

            new_fitness = func(new_harmony)

            better_indexes = new_fitness < self.fitness_memory
            self.harmony_memory[better_indexes] = new_harmony[better_indexes]
            self.fitness_memory[better_indexes] = new_fitness

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index]

# Usage:
budget = 1000
dim = 10
optimizer = ImprovedHarmonySearchOptimizer(budget, dim)
optimized_solution = optimizer(lambda x: np.sum(x ** 2))