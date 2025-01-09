import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = 10  # Harmony Memory Size
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.5  # Pitch Adjustment Rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.hms, self.dim))

        def generate_new_harmony(harmony_memory):
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[np.random.randint(self.hms), i]
                else:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.normal(0, 1)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory)
            harmony_memory = np.vstack((harmony_memory, new_harmony))
            harmony_memory = harmony_memory[np.argsort(func(harmony_memory))[:self.hms]]

        return harmony_memory[np.argmin(func(harmony_memory))]