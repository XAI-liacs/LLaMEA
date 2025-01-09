import numpy as np

class ImprovedHarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, bw_range=(0.01, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_range = bw_range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        
        for i in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    else:
                        k = np.random.randint(self.dim)
                        new_harmony[j] = harmony_memory[k] + np.random.uniform(-self.bw, self.bw)
                        new_harmony[j] = np.clip(new_harmony[j], func.bounds.lb, func.bounds.ub)  # Ensure within bounds
                
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = np.copy(new_harmony)
                
            # Adaptive bandwidth update
            self.bw *= np.exp(-0.01*i)  # Decrease bandwidth over time for better convergence
            
        return self.f_opt, self.x_opt