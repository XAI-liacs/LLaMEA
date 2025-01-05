import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_lower = 0.2
        self.F_upper = 0.8
        self.CR_lower = 0.2
        self.CR_upper = 1.0

    def adapt_parameters(self, it, max_it):
        self.F = self.F_upper - ((self.F_upper - self.F_lower) / max_it) * it
        self.CR = self.CR_lower + ((self.CR_upper - self.CR_lower) / max_it) * it

    def __call__(self, func):
        pop_size = 10 * self.dim
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(pop_size, self.dim))
        
        for i in range(self.budget):
            self.adapt_parameters(i, self.budget)
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial.copy()
                
                population[j] = trial if f < func(population[j]) else population[j]
        
        return self.f_opt, self.x_opt