import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))

        for _ in range(self.budget):
            new_population = np.zeros_like(population)
            for i in range(self.pop_size):
                indices = np.random.choice(range(self.pop_size), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt