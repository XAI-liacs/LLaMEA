import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.9, CR=0.8):
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
            for j in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(population[a] + self.F * (population[b] - population[c]), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[j])
                
                f = func(trial)
                if f < fitness[j]:
                    population[j] = trial
                    fitness[j] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt