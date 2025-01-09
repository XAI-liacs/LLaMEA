import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(20, self.budget // 5)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                       (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        num_evaluations = self.population_size

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.array([mutant[j] if np.random.rand() < self.CR else population[i, j] for j in range(self.dim)])
                
                # Apply a local random search to the trial vector
                local_search = trial + np.random.uniform(-0.1, 0.1, self.dim)
                local_search = np.clip(local_search, self.lower_bound, self.upper_bound)
                local_fitness = func(local_search)
                num_evaluations += 1

                # Selection
                trial_fitness = func(trial)
                num_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                # Adaptive learning using local search results
                if local_fitness < fitness[i]:
                    population[i] = local_search
                    fitness[i] = local_fitness
                
                if num_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]