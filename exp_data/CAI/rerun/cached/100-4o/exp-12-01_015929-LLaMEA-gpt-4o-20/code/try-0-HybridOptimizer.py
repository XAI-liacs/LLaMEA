import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0

    def evolve_population(self, func):
        F = 0.8  # Differential evolution step size
        CR = 0.9  # Crossover rate
        new_population = np.empty_like(self.population)

        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = self.population[indices]
            mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
            crossover_mask = np.random.rand(self.dim) < CR
            trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
            new_population[i] = trial_vector

            if self.used_budget < self.budget:
                trial_fitness = func(trial_vector)
                self.used_budget += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector

        return new_population

    def local_search(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            current_solution = self.population[i]
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)
            new_solution = np.clip(current_solution + perturbation, self.lower_bound, self.upper_bound)
            new_fitness = func(new_solution)
            self.used_budget += 1
            if new_fitness < self.fitness[i]:
                self.fitness[i] = new_fitness
                self.population[i] = new_solution

    def __call__(self, func):
        for _ in range(self.budget // self.population_size):
            if self.used_budget >= self.budget:
                break
            self.population = self.evolve_population(func)
            self.local_search(func)

        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]