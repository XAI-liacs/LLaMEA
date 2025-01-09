import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.Inf)
        self.cr = 0.5  # Crossover probability
        self.f = 0.8   # Differential weight

    def __call__(self, func):
        # Initialize population and evaluate fitness
        for i in range(self.population_size):
            self.fitness[i] = func(self.pop[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt, self.x_opt = self.fitness[i], self.pop[i]

        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                # Mutation and crossover
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = self.pop[idxs]
                mutant = x1 + self.f * (x2 - x3)
                mutant = np.clip(mutant, -5.0, 5.0)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.pop[i])
                f_trial = func(trial)
                evals += 1

                # Selection
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt, self.x_opt = f_trial, trial

                # Break if budget is reached
                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt