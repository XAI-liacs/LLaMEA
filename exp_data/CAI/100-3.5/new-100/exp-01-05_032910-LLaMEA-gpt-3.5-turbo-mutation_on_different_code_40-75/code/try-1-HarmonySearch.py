import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, bandwidth=0.01, pitch_adjust_rate=0.5, bandwidth_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth_decay = bandwidth_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, self.dim))
        for _ in range(self.budget):
            if np.random.rand() < self.pitch_adjust_rate:
                pitch_adjusted = np.clip(harmony_memory[np.random.randint(self.harmony_memory_size)] + np.random.uniform(-self.bandwidth, self.bandwidth), func.bounds.lb, func.bounds.ub)
            else:
                pitch_adjusted = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)

            f = func(pitch_adjusted)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = pitch_adjusted
                harmony_memory[np.argmax(harmony_memory)] = pitch_adjusted

            self.bandwidth *= self.bandwidth_decay

        return self.f_opt, self.x_opt