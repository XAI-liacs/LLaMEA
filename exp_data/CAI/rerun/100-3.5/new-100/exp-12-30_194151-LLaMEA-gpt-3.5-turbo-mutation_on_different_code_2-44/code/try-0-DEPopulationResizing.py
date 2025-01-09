import numpy as np

class DEPopulationResizing:
    def __init__(self, budget=10000, dim=10, population_size=50, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, func, population):
        new_population = []
        for i, target in enumerate(population):
            candidates = [ind for ind in population if ind is not target]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, target)
            if func(trial) < func(target):
                new_population.append(trial)
            else:
                new_population.append(target)
        return new_population

    def __call__(self, func):
        population = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.population_size)]
        for _ in range(self.budget // self.population_size):
            population = self.evolve_population(func, population)
            for ind in population:
                f = func(ind)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = ind

        return self.f_opt, self.x_opt