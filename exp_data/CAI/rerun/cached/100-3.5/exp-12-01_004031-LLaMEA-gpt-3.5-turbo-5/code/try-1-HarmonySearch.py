import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 20
        self.bandwidth = 0.01

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

        def improvise_new_harmony(harmony_memory):
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                else:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[idx, i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        harmony_values = np.array([func(h) for h in harmony_memory])
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = improvise_new_harmony(harmony_memory)
            new_value = func(new_harmony)
            if new_value < np.max(harmony_values):
                idx = np.argmax(harmony_values)
                harmony_memory[idx] = new_harmony
                harmony_values[idx] = new_value
        best_idx = np.argmin(harmony_values)
        return harmony_memory[best_idx]