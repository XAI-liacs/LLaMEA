import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.bandwidth = 0.01

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.harmony_memory_size, self.dim))

        def improvise_new_harmony(harmony_memory):
            new_harmony = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < 0.5:  # Pitch Adjustment
                    new_harmony[i] = new_harmony[i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = improvise_new_harmony(harmony_memory)
            if func(new_harmony) < func(harmony_memory[-1]):
                harmony_memory[-1] = new_harmony
                harmony_memory = harmony_memory[np.argsort([func(h) for h in harmony_memory])]

        return harmony_memory[0]