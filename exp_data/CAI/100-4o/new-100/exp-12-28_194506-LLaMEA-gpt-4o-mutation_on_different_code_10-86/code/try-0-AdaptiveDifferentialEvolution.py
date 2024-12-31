import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.pop = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.pop_fitness = np.full(self.population_size, np.Inf)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                x = self.pop[i]
                mutant = self.pop[a] + self.mutation_factor * (self.pop[b] - self.pop[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, x)

                f = func(trial)
                evaluations += 1

                if f < self.pop_fitness[i]:
                    self.pop[i] = trial
                    self.pop_fitness[i] = f

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt