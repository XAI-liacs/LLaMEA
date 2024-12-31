import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=100):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

            # Adaptive control of F and CR
            self.F = self.adaptive_parameter(self.F)
            self.CR = self.adaptive_parameter(self.CR)

            # Dynamic population size adjustment
            self.population_size = max(20, int(self.population_size * 0.99))

        return self.f_opt, self.x_opt

    def adaptive_parameter(self, param, lower=0.1, upper=0.9, rate=0.1):
        param += rate * (np.random.rand() - 0.5)
        return np.clip(param, lower, upper)