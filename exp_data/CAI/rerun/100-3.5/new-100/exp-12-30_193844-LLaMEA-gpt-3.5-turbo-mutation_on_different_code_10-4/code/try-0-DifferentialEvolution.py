import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, population, func):
        new_population = []
        for i, target in enumerate(population):
            candidates = [ind for ind in population if ind is not target]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            mutant = a + self.f * (b - c)
            trial = np.copy(target)
            for j in range(self.dim):
                if np.random.rand() < self.cr or j == np.random.randint(0, self.dim):
                    trial[j] = mutant[j]
            f_target = func(target)
            f_trial = func(trial)
            if f_trial < f_target:
                new_population.append(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = np.copy(trial)
            else:
                new_population.append(target)
        return new_population

    def __call__(self, func):
        population = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(10)]
        for _ in range(self.budget // 10):
            population = self.evolve_population(population, func)
        return self.f_opt, self.x_opt