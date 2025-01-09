import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hms=10, hmcr=0.95, par=0.5, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmonies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.hms, self.dim))
        f_values = np.apply_along_axis(func, 1, harmonies)
        
        for i in range(self.budget):
            new_harmony = np.copy(harmonies[np.argmin(f_values)])
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[j] = harmonies[np.random.randint(self.hms), j]
                    else:
                        new_harmony[j] = new_harmony[j] + np.random.uniform(-self.bw, self.bw)
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            new_f = func(new_harmony)
            if new_f < np.max(f_values):
                idx = np.argmax(f_values)
                harmonies[idx] = new_harmony
                f_values[idx] = new_f
                
            if new_f < self.f_opt:
                self.f_opt = new_f
                self.x_opt = new_harmony
        
        return self.f_opt, self.x_opt