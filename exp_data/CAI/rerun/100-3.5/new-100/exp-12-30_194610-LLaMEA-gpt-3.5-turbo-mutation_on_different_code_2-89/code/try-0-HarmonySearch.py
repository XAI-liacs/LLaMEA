import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.01, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Bandwidth
        self.f_opt = np.Inf
        self.x_opt = None
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (dim,))
        
    def __call__(self, func):
        for i in range(self.budget):
            new_harmony = np.copy(self.harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = self.harmony_memory[np.random.randint(self.dim)]  # Memory consideration
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-self.bw, self.bw)  # Pitch adjustment

            new_harmony = np.clip(new_harmony, -5.0, 5.0)
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                self.harmony_memory = new_harmony
                
        return self.f_opt, self.x_opt