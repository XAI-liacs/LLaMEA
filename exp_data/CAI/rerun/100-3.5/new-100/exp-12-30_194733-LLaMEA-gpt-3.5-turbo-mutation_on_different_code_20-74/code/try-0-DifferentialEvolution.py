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
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = list(range(self.budget))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                trial_vector = population[a] + self.F * (population[b] - population[c])
                mask = np.random.rand(self.dim) < self.CR
                target_vector = np.where(mask, trial_vector, population[i])
                
                f = func(target_vector)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = target_vector
                    population[i] = target_vector
                
        return self.f_opt, self.x_opt