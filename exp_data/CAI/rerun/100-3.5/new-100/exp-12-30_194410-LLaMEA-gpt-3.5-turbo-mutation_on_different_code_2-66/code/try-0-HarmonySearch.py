import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, pitch_adjustment_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim))
        
        for i in range(self.budget):
            if np.random.rand() < self.pitch_adjustment_rate:
                x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            else:
                x = harmony_memory[np.random.randint(self.harmony_memory_size)]
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
                harmony_memory[np.argmax(harmony_memory[:, 1])] = x
            
        return self.f_opt, self.x_opt