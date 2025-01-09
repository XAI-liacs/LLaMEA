import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate fitness
            for i in range(self.population_size):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    self.evaluations += 1
                    if self.evaluations >= self.budget:
                        break

            if self.evaluations >= self.budget:
                break

            # Differential Evolution with Adaptive Gaussian Mutation
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.scaling_factor * (b - c)

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, self.population[i])

                # Adaptive Gaussian Mutation
                mutation_strength = np.random.normal(0, 0.1, self.dim) * (self.upper_bound - self.lower_bound)
                trial = np.clip(trial + mutation_strength, self.lower_bound, self.upper_bound)

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

        best_index = np.argmin(self.fitness)
        return self.population[best_index]