import numpy as np

class JADE_DE:
    def __init__(self, budget=10000, dim=10, p=0.1, c=0.1, p_best=0.05):
        self.budget = budget
        self.dim = dim
        self.p = p
        self.c = c
        self.p_best = p_best
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            idxs = np.random.choice(range(self.budget), 3, replace=False)
            x_r1, x_r2, x_r3 = population[idxs]
            x_new = population[i] + self.c * (x_r1 - population[i]) + self.p * (x_r2 - x_r3)
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            
            f_new = func(x_new)
            if f_new < fitness[i]:
                population[i] = x_new
                fitness[i] = f_new
                
                if np.random.rand() < self.p_best:
                    idx_best = np.argsort(fitness)[0]
                    population[i] = population[i] + np.random.normal(0, 1, self.dim) * (population[idx_best] - population[i])
            
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = population[i]
                
        return self.f_opt, self.x_opt