import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.bandwidth = 0.01
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.harmony_memory_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            new_harmony = np.clip(np.random.uniform(-self.bandwidth, self.bandwidth, size=self.dim) + np.random.choice(self.harmony_memory), self.lower_bound, self.upper_bound)
            if func(new_harmony) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
        return self.harmony_memory[0]