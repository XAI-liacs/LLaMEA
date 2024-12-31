import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20, F_min=0.1, F_max=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_min = F_min
        self.F_max = F_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), size=3, replace=False)
                a, b, c = population[idxs]
                F_current = np.clip(np.random.normal(self.F, 0.1), self.F_min, self.F_max)  # Dynamic F adjustment
                mutant = population[a] + F_current * (population[b] - population[c])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f = func(trial)
                if f < func(population[j]):
                    population[j] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                    
        return self.f_opt, self.x_opt