import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                indices = [idx for idx in range(self.budget) if idx != j]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                trial = np.copy(population[j])
                mask = np.random.rand(self.dim) < self.CR
                
                if not np.any(mask):
                    mask[np.random.randint(0, self.dim)] = True
                
                trial[mask] = mutant[mask]
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
        return self.f_opt, self.x_opt