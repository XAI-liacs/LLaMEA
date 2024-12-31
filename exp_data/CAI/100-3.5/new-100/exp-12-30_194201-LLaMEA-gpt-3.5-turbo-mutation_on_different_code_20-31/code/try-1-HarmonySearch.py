import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, bw_min=0.001, bw_max=0.1, bw_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.bw_decay = bw_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        
        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[d] = harmony_memory[d]
                else:
                    new_harmony[d] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < self.par:
                    new_harmony[d] += np.random.uniform(-self.bw, self.bw)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony
            
            self.bw = max(self.bw_min, self.bw * self.bw_decay) if i % 100 == 0 else self.bw  # Adaptive bandwidth control
        
        return self.f_opt, self.x_opt