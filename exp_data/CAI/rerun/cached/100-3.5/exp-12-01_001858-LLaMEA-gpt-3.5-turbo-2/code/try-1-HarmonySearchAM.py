import numpy as np

class HarmonySearchAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_rate = 0.7
        self.pitch_adjust_rate = 0.5
        self.bandwidth = (self.upper_bound - self.lower_bound) * 0.01
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (int(self.harmony_memory_rate * budget), dim))

    def __call__(self, func):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        for i in range(self.budget):
            if np.random.rand() < self.harmony_memory_rate:
                index = np.random.randint(len(self.harmony_memory))
                for j in range(self.dim):
                    if np.random.rand() < self.pitch_adjust_rate:
                        self.bandwidth = np.maximum(self.bandwidth * 0.9, 1e-8)  # Self-adaptive bandwidth adjustment
                        harmonies[i, j] = self.harmony_memory[index, j] + np.random.uniform(-self.bandwidth, self.bandwidth)
            harmonies[i] = np.clip(harmonies[i], self.lower_bound, self.upper_bound)
            if func(harmonies[i]) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = harmonies[i]
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
        return self.harmony_memory[0]