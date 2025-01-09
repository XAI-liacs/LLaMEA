import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = population_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover rate
        self.f_opt = np.Inf
        self.x_opt = None
        self.success_rate = 0.5

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size

        for _ in range(self.budget // self.pop_size):
            successes = 0
            for i in range(self.pop_size):
                idxs = list(range(self.pop_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Adaptive mutation factor
                F_dynamic = self.F * (1 - (_ / (self.budget // self.pop_size)))
                mutation_factor = F_dynamic + 0.1 * np.random.randn(self.dim)

                mutant = np.clip(a + mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < (self.CR * self.success_rate)
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    successes += 1
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            self.success_rate = 0.5 * (self.success_rate + successes / self.pop_size)

        return self.f_opt, self.x_opt