import numpy as np

class HarmonySearchRefined:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, bandwidth=0.01, global_bandwidth=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.global_bandwidth = global_bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        
        for _ in range(self.budget):
            new_harmony = np.random.uniform(-5.0, 5.0, self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.bandwidth:
                    new_harmony[i] = np.random.choice(harmony_memory[:, i])
                elif np.random.rand() < self.global_bandwidth:
                    new_harmony[i] = np.random.uniform(-5.0, 5.0)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
            idx = np.argmax(harmony_memory, axis=0)
            harmony_memory[idx] = new_harmony if f < func(harmony_memory[idx]) else harmony_memory[idx]
        
        return self.f_opt, self.x_opt