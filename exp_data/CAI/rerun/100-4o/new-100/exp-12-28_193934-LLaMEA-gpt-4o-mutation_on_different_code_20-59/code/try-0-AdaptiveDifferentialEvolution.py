import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evals = self.pop_size
        while evals < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adaptive parameter adjustment
            self.F = np.random.uniform(0.5, 1.0)
            self.CR = np.random.uniform(0.8, 1.0)

        return self.f_opt, self.x_opt