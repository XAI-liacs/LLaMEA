import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                
                min_idx = np.argmin(fitness)
                if f_trial < fitness[min_idx]:
                    population[min_idx] = trial
                    fitness[min_idx] = f_trial
                    
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt