import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, CR=0.8, F=0.5, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.CR = CR
        self.F = F
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial
                
        return self.f_opt, self.x_opt