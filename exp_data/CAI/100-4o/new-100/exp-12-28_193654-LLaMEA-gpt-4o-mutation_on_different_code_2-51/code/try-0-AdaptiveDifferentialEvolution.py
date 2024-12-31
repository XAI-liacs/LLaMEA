import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, f_min=0.1, f_max=0.9, cr_min=0.1, cr_max=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_min = f_min
        self.f_max = f_max
        self.cr_min = cr_min
        self.cr_max = cr_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F = np.random.uniform(self.f_min, self.f_max)
                CR = np.random.uniform(self.cr_min, self.cr_max)
                
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                f = func(trial)
                evaluations += 1

                if f < fitness[i]:
                    pop[i], fitness[i] = trial, f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt