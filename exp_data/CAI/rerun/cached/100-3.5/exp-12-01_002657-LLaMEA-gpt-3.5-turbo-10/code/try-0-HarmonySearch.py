import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=20, bandwidth=0.01, pitch_adjust_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(h) for h in harmony_memory])

        for _ in range(self.budget):
            # Create a new harmony
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    index = np.random.randint(self.harmony_memory_size)
                    new_harmony[d] = harmony_memory[index, d] + np.random.uniform(-self.bandwidth, self.bandwidth)
                else:
                    new_harmony[d] = np.random.uniform(-5.0, 5.0)

            new_fitness = func(new_harmony)
            # Update harmony memory if new harmony is better
            if new_fitness < harmony_memory_fitness.max():
                index = np.argmax(harmony_memory_fitness)
                harmony_memory[index] = new_harmony
                harmony_memory_fitness[index] = new_fitness

        best_index = np.argmin(harmony_memory_fitness)
        best_solution = harmony_memory[best_index]
        return best_solution