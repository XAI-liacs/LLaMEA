import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Common practice is 5 to 10 times the dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate
        self.shrink_factor = 0.99  # Factor to adjust F and CR over time

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                # Step 1: Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Step 2: Crossover
                crossover_points = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_points, mutant, population[i])

                # Step 3: Selection
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Check if budget is exhausted
                if evals >= self.budget:
                    break

            # Adaptive mechanism
            self.F *= self.shrink_factor
            self.CR *= self.shrink_factor

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]