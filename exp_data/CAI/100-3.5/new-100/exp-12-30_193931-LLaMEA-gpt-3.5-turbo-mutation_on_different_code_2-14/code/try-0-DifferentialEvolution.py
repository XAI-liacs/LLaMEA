import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, func, population):
        new_population = []
        for i, target in enumerate(population):
            a, b, c = np.random.choice(population, 3, replace=False)
            mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
            trial = np.where(np.random.rand(self.dim) < self.CR, mutant, target)
            f_trial = func(trial)
            if f_trial < func(target):
                new_population.append(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            else:
                new_population.append(target)
        return new_population

    def __call__(self, func):
        population = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.pop_size)]
        for _ in range(self.budget // self.pop_size):
            population = self.evolve_population(func, population)
        return self.f_opt, self.x_opt