import numpy as np

class HybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // 10)
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.num_evaluations = 0

    def __call__(self, func):
        for i in range(self.population_size):
            if self.num_evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.num_evaluations += 1
            else:
                break

        while self.num_evaluations < self.budget:
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break

                # Mutation and Crossover (Differential Evolution)
                a, b, c = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.population[i])

                # Evaluate Trial Vector
                trial_fitness = func(trial)
                self.num_evaluations += 1

                # Adaptive Local Search
                if trial_fitness < self.fitness[i]:
                    step_size = 0.1 * (self.upper_bound - self.lower_bound)
                    local_search_vector = trial + step_size * np.random.uniform(-1, 1, self.dim)
                    local_search_vector = np.clip(local_search_vector, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_search_vector)
                    self.num_evaluations += 1
                    if local_fitness < trial_fitness:
                        trial, trial_fitness = local_search_vector, local_fitness

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

        best_index = np.argmin(self.fitness)
        return self.population[best_index]