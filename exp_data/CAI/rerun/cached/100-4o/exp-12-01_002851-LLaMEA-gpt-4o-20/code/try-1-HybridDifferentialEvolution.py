import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.current_evaluations = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, 
                                 (self.population_size, self.dim))

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break

                # Adaptive Differential Evolution mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                adaptive_factor = self.mutation_factor + 0.2 * np.random.rand()
                mutant = x0 + adaptive_factor * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Adaptive Crossover
                crossover = np.random.rand(self.dim) < (self.crossover_rate + 0.1 * np.random.rand())
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial
                trial_fitness = func(trial)
                self.current_evaluations += 1

                # Selection and Local Search
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                    # Local Search
                    local_trial = trial + np.random.normal(0, 0.1, self.dim)
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    self.current_evaluations += 1
                    if local_fitness < trial_fitness:
                        population[i] = local_trial
                        fitness[i] = local_fitness
                        if local_fitness < best_fitness:
                            best_solution = local_trial
                            best_fitness = local_fitness

        return best_solution