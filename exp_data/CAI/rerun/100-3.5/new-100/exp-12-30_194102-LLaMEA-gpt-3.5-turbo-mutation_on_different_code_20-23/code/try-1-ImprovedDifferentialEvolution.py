import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.8, CR_init=0.9, F_decay=0.99, CR_increase=0.02):
        self.budget = budget
        self.dim = dim
        self.F = F_init  # Differential weight
        self.CR = CR_init  # Crossover probability
        self.F_decay = F_decay  # Decay rate for F
        self.CR_increase = CR_increase  # Increase rate for CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(len(population)):
                idxs = np.random.choice(np.delete(np.arange(len(population)), j, axis=0), 3, replace=False)
                a, b, c = population[idxs]
                
                mutant = population[j] + self.F * (a - b)
                mask = np.random.rand(self.dim) < self.CR
                trial = np.where(mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]
            
            self.F *= self.F_decay
            self.CR = min(1.0, self.CR + self.CR_increase)
        
        return self.f_opt, self.x_opt