import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        memory = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(self.dim,))
        
        for i in range(self.budget):
            new_solution = np.copy(memory)
            
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[j] = memory[j]
                else:
                    new_solution[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    
                    if np.random.rand() < self.par:
                        rand_index = np.random.randint(self.dim)
                        new_solution[rand_index] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution
                memory = np.copy(new_solution)
            
        return self.f_opt, self.x_opt