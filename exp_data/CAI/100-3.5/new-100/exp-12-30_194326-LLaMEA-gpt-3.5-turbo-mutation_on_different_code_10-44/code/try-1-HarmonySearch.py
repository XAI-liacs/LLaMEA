import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, bandwidth=0.01, bandwidth_reduction_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.bandwidth_reduction_factor = bandwidth_reduction_factor
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, self.dim))
        
        for i in range(self.budget):
            new_bandwidth = self.bandwidth * self.bandwidth_reduction_factor**i
            new_harmony = np.clip(harmony_memory.mean(axis=0) + np.random.uniform(-new_bandwidth, new_bandwidth, size=self.dim), func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory[np.argmax(func(harmony_memory))] = new_harmony
            
        return self.f_opt, self.x_opt