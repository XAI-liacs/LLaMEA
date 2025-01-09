import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim))
        
        for _ in range(self.budget):
            x_new = np.clip(np.mean(harmony_memory, axis=0) + np.random.uniform(-self.bandwidth, self.bandwidth, self.dim),
                            func.bounds.lb, func.bounds.ub)
            
            f_new = func(x_new)
            
            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new
                
                worst_idx = np.argmax([func(x) for x in harmony_memory])
                harmony_memory[worst_idx] = x_new
                
        return self.f_opt, self.x_opt