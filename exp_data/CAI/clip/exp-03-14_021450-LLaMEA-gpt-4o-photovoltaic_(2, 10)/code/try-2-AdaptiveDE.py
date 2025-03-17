import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = np.random.rand(self.population_size, self.dim)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.5

    def select_parents(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        np.random.shuffle(indices)
        return indices[:3]

    def mutate(self, idx):
        a, b, c = self.select_parents(idx)
        mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        lower_bound, upper_bound = self.bounds
        return np.clip(mutant_vector, lower_bound, upper_bound)

    def crossover(self, target, mutant):
        crossover_vector = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.crossover_rate or i == np.random.randint(self.dim):
                crossover_vector[i] = mutant[i]
        return crossover_vector

    def optimize(self, func):
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                target_vector = self.population[i]
                mutant_vector = self.mutate(i)
                trial_vector = self.crossover(target_vector, mutant_vector)
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector

                if trial_fitness < func(target_vector):
                    self.population[i] = trial_vector

                # Dynamically adjust mutation and crossover rates
                self.mutation_factor = 0.5 + 0.2 * (self.best_fitness / trial_fitness)
                self.crossover_rate = 0.9 - 0.3 * (self.best_fitness / trial_fitness)

        return self.best_solution, self.best_fitness

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.population = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * np.random.rand(self.population_size, self.dim)
        return self.optimize(func)