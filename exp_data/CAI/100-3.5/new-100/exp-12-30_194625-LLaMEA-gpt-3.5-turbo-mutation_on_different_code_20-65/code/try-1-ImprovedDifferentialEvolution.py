import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = population[idxs]
                
                F = self.F + np.random.normal(0, 0.1)  # Adaptive control for F
                F = max(0.1, min(F, 0.9))
                
                CR = self.CR + np.random.normal(0, 0.1)  # Adaptive control for CR
                CR = max(0.1, min(CR, 1.0))
                
                mutant = a + F * (b - c)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial
            
        return self.f_opt, self.x_opt