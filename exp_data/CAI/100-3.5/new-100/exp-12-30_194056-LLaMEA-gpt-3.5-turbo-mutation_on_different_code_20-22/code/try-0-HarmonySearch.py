import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.9, par=0.5, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Bandwidth
        self.f_opt = np.Inf
        self.x_opt = None
        self.memory = np.random.uniform(-5.0, 5.0, (dim,))
    
    def __call__(self, func):
        for _ in range(self.budget):
            new_harmony = np.copy(self.memory)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = self.memory[i]
                else:
                    new_harmony[i] = np.random.uniform(-5.0, 5.0)
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-self.bw, self.bw)
            
            new_harmony_fitness = func(new_harmony)
            if new_harmony_fitness < self.f_opt:
                self.f_opt = new_harmony_fitness
                self.x_opt = new_harmony
                self.memory = new_harmony
                
        return self.f_opt, self.x_opt