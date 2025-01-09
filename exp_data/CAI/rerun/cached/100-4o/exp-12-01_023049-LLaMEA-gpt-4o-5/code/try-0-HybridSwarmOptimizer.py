import numpy as np

class HybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.pa = 0.25  # Probability for Cuckoo Search
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Differential Evolution Step
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            # Cuckoo Search Step
            for i in range(self.population_size):
                if np.random.rand() < self.pa:
                    levy_flight = np.random.standard_normal(self.dim) * 0.01
                    cuckoo = population[i] + levy_flight * (population[i] - self.best_solution)
                    cuckoo = np.clip(cuckoo, self.lower_bound, self.upper_bound)

                    cuckoo_fitness = func(cuckoo)
                    evaluations += 1

                    if cuckoo_fitness < fitness[i]:
                        population[i] = cuckoo
                        fitness[i] = cuckoo_fitness

                        if cuckoo_fitness < self.best_fitness:
                            self.best_solution = cuckoo
                            self.best_fitness = cuckoo_fitness

                if evaluations >= self.budget:
                    break

        return self.best_solution