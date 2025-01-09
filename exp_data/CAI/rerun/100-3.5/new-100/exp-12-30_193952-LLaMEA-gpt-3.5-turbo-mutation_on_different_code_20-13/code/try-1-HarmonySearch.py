import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, adapt_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        harmony_memory = np.random.uniform(lower_bound, upper_bound, (self.dim,))
        
        for _ in range(self.budget):
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[i] = np.random.uniform(lower_bound, upper_bound)
                    else:
                        rand_idx = np.random.randint(self.dim)
                        new_harmony[i] = harmony_memory[rand_idx] + np.random.uniform(-self.bw, self.bw)
                        self.bw *= self.adapt_rate  # Adaptive bandwidth adjustment
            
            f = func(new_harmony)
            if f < func(harmony_memory):
                harmony_memory = np.copy(new_harmony)
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = np.copy(new_harmony)
                
        return self.f_opt, self.x_opt