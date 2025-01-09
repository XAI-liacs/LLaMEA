import numpy as np

class ADEBanditOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.scaling_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation with dynamic scaling factor adjustment
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                dynamic_scaling_factor = self.scaling_factor + np.random.uniform(-0.1, 0.1)
                mutant = np.clip(x0 + dynamic_scaling_factor * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Early stopping if budget is exhausted
                if eval_count >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]