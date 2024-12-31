import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, bandwidth=0.01, bandwidth_reduction=0.9):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.bandwidth_reduction = bandwidth_reduction
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, self.dim))
        
        for _ in range(self.budget):
            if np.random.rand() < 0.7:  # Memory consideration
                x = harmony_memory[np.random.choice(self.harmony_memory_size)]
            else:  # Pitch adjustment
                x = np.clip(x + np.random.uniform(-self.bandwidth, self.bandwidth), func.bounds.lb, func.bounds.ub)
                
                # Adaptive bandwidth adjustment
                self.bandwidth *= self.bandwidth_reduction
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
                # Update harmony memory
                idx = np.argmax(np.array([func(x) for x in harmony_memory]))
                if f < func(harmony_memory[idx]):
                    harmony_memory[idx] = x
            
        return self.f_opt, self.x_opt