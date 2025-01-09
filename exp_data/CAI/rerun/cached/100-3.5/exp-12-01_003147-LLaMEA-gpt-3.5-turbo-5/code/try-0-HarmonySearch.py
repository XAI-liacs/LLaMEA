import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_size = 10
        self.hm_accept_rate = 0.95
        self.bandwidth = 0.01
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))

    def generate_new_harmony(self):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hm_accept_rate:
                new_harmony[i] = self.memory[np.random.randint(self.hm_size), i]
            else:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return new_harmony

    def pitch_adjustment(self, harmony):
        new_harmony = np.copy(harmony)
        for i in range(self.dim):
            if np.random.rand() < 0.3:
                new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony

    def __call__(self, func):
        for _ in range(self.budget):
            new_harmony = self.generate_new_harmony()
            new_harmony = self.pitch_adjustment(new_harmony)
            if func(new_harmony) < func(self.memory[0]):
                self.memory[0] = new_harmony
        return self.memory[0]