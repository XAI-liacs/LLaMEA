import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, pitch_adjust_rate=0.1, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, self.dim))
        
        for _ in range(self.budget):
            if np.random.rand() < self.pitch_adjust_rate:
                idx = np.random.randint(self.harmony_memory_size)
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        harmony_memory[idx, j] = np.clip(harmony_memory[idx, j] + np.random.uniform(-self.bandwidth, self.bandwidth), func.bounds.lb, func.bounds.ub)
            
            f_values = np.array([func(x) for x in harmony_memory])
            best_idx = np.argmin(f_values)
            if f_values[best_idx] < self.f_opt:
                self.f_opt = f_values[best_idx]
                self.x_opt = harmony_memory[best_idx]
            
            self.bandwidth *= 0.999  # Adaptive bandwidth adjustment

        return self.f_opt, self.x_opt