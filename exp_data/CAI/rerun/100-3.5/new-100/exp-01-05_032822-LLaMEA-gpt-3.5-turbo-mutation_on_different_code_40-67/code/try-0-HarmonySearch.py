import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, pitch_adjust_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        hm = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim))
        for i in range(self.budget):
            new_harmony = np.mean(hm, axis=0)
            for j in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                hm[np.argmax(hm)] = new_harmony
            
        return self.f_opt, self.x_opt