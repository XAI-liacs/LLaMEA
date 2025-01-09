import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * self.dim  # Using a scale factor based on dimensionality
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        rng = np.random.default_rng()
        population = rng.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                indices = rng.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)
                cross_points = rng.random(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                if rng.random() < 0.2:  # Adaptive local search with 20% probability
                    local_search = rng.uniform(-0.1, 0.1, self.dim)
                    trial = np.clip(trial + local_search, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]