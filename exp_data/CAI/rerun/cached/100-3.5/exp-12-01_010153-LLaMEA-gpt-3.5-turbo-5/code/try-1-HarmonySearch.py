import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.bandwidth = 0.01

    def generate_initial_harmony_memory(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

    def improvise_new_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < 0.8:  # Consider memory
                new_harmony[np.random.randint(self.harmony_memory_size), i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
            else:  # Pitch adjustment
                new_harmony[np.random.randint(self.harmony_memory_size), i] = np.clip(harmony_memory[np.random.randint(self.harmony_memory_size), i] + np.random.uniform(-self.bandwidth, self.bandwidth), self.lower_bound, self.upper_bound)
        return new_harmony

    def __call__(self, func):
        harmony_memory = self.generate_initial_harmony_memory()
        for _ in range(self.budget):
            new_harmony = self.improvise_new_harmony(harmony_memory)
            if func(new_harmony) < func(harmony_memory[np.argmin(func(harmony_memory))]):
                harmony_memory[np.argmin(func(harmony_memory))] = new_harmony
        return harmony_memory[np.argmin(func(harmony_memory))]