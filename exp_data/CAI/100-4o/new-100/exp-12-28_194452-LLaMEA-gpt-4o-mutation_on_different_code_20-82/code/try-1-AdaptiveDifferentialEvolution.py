import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * self.dim
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.elite_fraction = 0.1  # Fraction of elite solutions to retain

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        evaluations = self.pop_size

        while evaluations < self.budget:
            trial_population = np.copy(population)
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                self.f = 0.5 + np.random.rand() * 0.5  # Self-adaptive F
                self.cr = 0.5 + np.random.rand() * 0.5  # Self-adaptive CR

                mutant_vector = np.clip(a + self.f * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(cross_points, mutant_vector, population[i])
                trial_population[i] = trial_vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial_vector

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

            elite_count = int(self.elite_fraction * self.pop_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_population = population[elite_indices]
            population = np.concatenate((elite_population, population[elite_count:]))

        return self.f_opt, self.x_opt