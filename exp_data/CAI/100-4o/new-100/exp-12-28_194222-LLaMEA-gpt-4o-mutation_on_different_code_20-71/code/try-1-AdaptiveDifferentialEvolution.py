import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20  # A common choice is 10 times the dimensionality
        self.mutation_factor = 0.5  # Initial differential weight
        self.crossover_rate = 0.7  # Initial crossover probability
        self.scale_factor_adaptation = 0.1  # Adaptation rate for mutation factor
        self.crossover_rate_adaptation = 0.05  # Adaptation rate for crossover rate
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        while self.budget > 0:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    self.mutation_factor = min(1.0, self.mutation_factor + self.scale_factor_adaptation)
                    self.crossover_rate = min(1.0, self.crossover_rate + self.crossover_rate_adaptation)
                else:
                    new_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.scale_factor_adaptation)
                    self.crossover_rate = max(0.5, self.crossover_rate - self.crossover_rate_adaptation)

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if self.budget <= 0:
                    break

            elite_idx = np.argmin(fitness)
            new_population[0] = population[elite_idx]  # Elitism strategy
            population[:] = new_population

        return self.f_opt, self.x_opt