import numpy as np

class HarmonySearchImproved:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01, bw_range=(0.01, 0.5)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_range = bw_range
        self.f_opt = np.Inf
        self.x_opt = None
        
    def improvisation(self, func, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                if np.random.rand() < self.par:
                    new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                else:
                    index = np.random.choice(len(harmony_memory))
                    new_harmony[i] = harmony_memory[index]
            else:
                bw_adjusted = self.bw * (self.bw_range[1] - self.bw) / (self.bw_range[1] - self.bw_range[0])  # Adaptive pitch adjustment
                new_harmony[i] += np.random.uniform(-bw_adjusted, bw_adjusted)
                new_harmony[i] = np.clip(new_harmony[i], func.bounds.lb, func.bounds.ub)
        return new_harmony
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        for _ in range(self.budget):
            new_harmony = self.improvisation(func, harmony_memory)
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = new_harmony
        return self.f_opt, self.x_opt