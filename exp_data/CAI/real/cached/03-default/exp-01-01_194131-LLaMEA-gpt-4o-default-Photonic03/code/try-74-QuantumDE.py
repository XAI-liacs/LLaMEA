import numpy as np

class QuantumDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.crossover_rate = 0.5
        self.mutation_factor = 0.8

    def _quantum_behavior(self, candidate, leader):
        distance = candidate - leader
        q_bit = np.random.normal(0, 1, self.dim)
        step = self.mutation_factor * distance * q_bit
        return candidate + step

    def _select_parents(self):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        return self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def _mutate(self, a, b, c):
        mutant_vector = a + self.mutation_factor * (b - c)
        return np.clip(mutant_vector, 0, 1)

    def _adapt_parameters(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.crossover_rate = np.clip(np.random.normal(self.crossover_rate, 0.1), 0.3, 0.9)
            self.mutation_factor = np.clip(np.random.normal(self.mutation_factor, 0.1), 0.5, 1.0)

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_solution = self.population[i]
                self.best_fitness = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_solution

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = self._select_parents()
                mutant = self._mutate(a, b, c)
                offspring = self._crossover(self.population[i], mutant)
                offspring = self._quantum_behavior(offspring, self.best_solution)

                offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)
                offspring_fitness = func(offspring)

                if offspring_fitness < self.fitness[i]:
                    self.population[i] = offspring
                    self.fitness[i] = offspring_fitness

                    if offspring_fitness < self.best_fitness:
                        self.best_solution = offspring
                        self.best_fitness = offspring_fitness

                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

            self._adapt_parameters()

        return self.best_solution