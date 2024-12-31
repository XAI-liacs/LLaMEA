import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, bw_range=(0.01, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Bandwidth
        self.bw_range = bw_range  # Bandwidth Range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        
        for _ in range(self.budget):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(self.budget)
                x = harmony_memory[idx]
            else:
                rand_idx = np.random.choice(np.arange(self.budget))
                x = harmony_memory[rand_idx] + np.random.uniform(-self.bw, self.bw, self.dim)

            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            f = func(x)
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
                harmony_memory[np.argmax(func(harmony_memory))] = x
            
            self.bw = np.clip(self.bw * (1 - 0.1/self.budget), self.bw_range[0], self.bw_range[1])
            
        return self.f_opt, self.x_opt