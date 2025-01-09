import numpy as np

class AdaptiveDECrowding:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(dim * 2)
        self.scale_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        func_evals = self.population_size

        # Optimization loop
        while func_evals < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break

                # Mutation
                candidates = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[candidates]
                mutant = np.clip(x1 + self.scale_factor * (x2 - x3), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_probability or j == j_rand:
                        trial[j] = mutant[j]

                # Selection with crowding
                trial_fitness = func(trial)
                func_evals += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

            population = new_population

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]