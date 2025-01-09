import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.bandwidth = 0.1

    def generate_initial_harmony_memory(self):
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

    def generate_new_harmony(self):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            else:
                rand_idx = np.random.randint(self.harmony_memory_size)
                new_harmony[i] = self.harmony_memory[rand_idx, i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony

    def __call__(self, func):
        self.generate_initial_harmony_memory()
        for _ in range(self.budget):
            new_harmony = self.generate_new_harmony()
            if func(new_harmony) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony
                self.harmony_memory = np.sort(self.harmony_memory, axis=0)
        return self.harmony_memory[0]