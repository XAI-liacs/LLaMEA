import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hmcr = 0.95  # Harmony Memory Consideration Rate
        self.par = 0.3  # Pitch Adjustment Rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_harmony(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def __call__(self, func):
        harmony_memory = [self.generate_harmony() for _ in range(self.budget)]
        for _ in range(self.budget):
            new_harmony = []
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony.append(harmony_memory[np.random.randint(0, len(harmony_memory))][i])
                else:
                    new_harmony.append(np.random.uniform(self.lower_bound, self.upper_bound))
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-1, 1) * (self.upper_bound - self.lower_bound) * np.exp(-2.0 * _ / self.budget)
            if func(new_harmony) < func(harmony_memory[np.argmax([func(h) for h in harmony_memory])]):
                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony
        return harmony_memory[np.argmin([func(h) for h in harmony_memory])]