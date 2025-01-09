import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        self.CR_min = 0.1  # Minimum CR
        self.CR_max = 0.9  # Maximum CR
        self.F_min = 0.2   # Minimum F
        self.F_max = 0.8   # Maximum F
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                self.CR = np.clip(np.random.normal(self.CR, 0.1), self.CR_min, self.CR_max)
                self.F = np.clip(np.random.normal(self.F, 0.1), self.F_min, self.F_max)
                
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = j
                        
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        return self.f_opt, self.x_opt