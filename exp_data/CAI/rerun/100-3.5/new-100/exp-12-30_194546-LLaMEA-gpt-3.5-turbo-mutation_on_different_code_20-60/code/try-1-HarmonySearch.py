import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Bandwidth for pitch adjustment
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        
        for i in range(self.budget):
            new_solution = np.copy(harmony_memory)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[j] = harmony_memory[j]
                else:
                    new_solution[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    
                if np.random.rand() < self.par:
                    new_solution[j] += np.random.uniform(-self.bw, self.bw)
            
            f = func(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution
                if np.random.rand() < self.hmcr:
                    harmony_memory = new_solution  # Adaptive HMCR
        return self.f_opt, self.x_opt