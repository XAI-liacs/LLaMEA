import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def generate_new_harmony(harmony_memory):
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:  # Harmony Memory Consideration Rate
                    if np.random.rand() < self.par:  # Pitch Adjustment Rate
                        rand_index = np.random.randint(self.budget)
                        new_harmony[rand_index, i] += np.random.uniform(-self.bw, self.bw)
                        new_harmony[rand_index, i] = np.clip(new_harmony[rand_index, i], -5.0, 5.0)
                    else:
                        new_harmony[np.random.randint(self.budget), i] = np.random.uniform(-5.0, 5.0)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory)
            harmony_memory = np.vstack((harmony_memory, new_harmony))
            harmony_memory = harmony_memory[np.argsort(func(harmony_memory))[:self.budget]]

        return harmony_memory[np.argmin(func(harmony_memory))]