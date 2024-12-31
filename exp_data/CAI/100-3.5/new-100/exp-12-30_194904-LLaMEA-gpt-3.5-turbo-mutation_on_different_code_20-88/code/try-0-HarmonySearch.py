import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        for i in range(self.budget):
            new_solution = np.zeros((self.dim,))
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[j] = harmony_memory[j]
                else:
                    rand_index = np.random.randint(self.dim)
                    new_solution[j] = harmony_memory[rand_index]
                    if np.random.rand() < self.par:
                        new_solution[j] = new_solution[j] + self.bw * np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution
                harmony_memory = np.copy(new_solution)
            
        return self.f_opt, self.x_opt