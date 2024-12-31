import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, bandwidth=0.01, pitch_adjust_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim))
        
        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    index = np.random.choice(self.harmony_memory_size)
                    new_harmony[j] = harmony_memory[index, j] + np.random.uniform(-self.bandwidth, self.bandwidth)
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
            worst_index = np.argmax([func(h) for h in harmony_memory])
            if f < func(harmony_memory[worst_index]):
                harmony_memory[worst_index] = new_harmony
            
        return self.f_opt, self.x_opt