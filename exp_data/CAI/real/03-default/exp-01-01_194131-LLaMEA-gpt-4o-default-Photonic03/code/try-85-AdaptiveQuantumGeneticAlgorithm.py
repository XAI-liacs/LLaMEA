import numpy as np

class AdaptiveQuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7

    def quantum_crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def quantum_mutation(self, offspring):
        mutation_vector = np.random.normal(0, 1, self.dim)
        offspring += self.mutation_rate * mutation_vector
        return offspring

    def _evaluate(self, individual, func):
        fitness = func(individual)
        return fitness

    def _select_parents(self):
        indices = np.random.choice(self.population_size, 2, replace=False)
        return self.population[indices[0]], self.population[indices[1]]

    def _update_population(self, func, bounds):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self._select_parents()
            if np.random.rand() < self.crossover_rate:
                child1, child2 = self.quantum_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = self.quantum_mutation(child1)
            child2 = self.quantum_mutation(child2)

            child1 = np.clip(child1, bounds.lb, bounds.ub)
            child2 = np.clip(child2, bounds.lb, bounds.ub)

            new_population.extend([child1, child2])

        self.population = np.array(new_population[:self.population_size])

    def _adapt_parameters(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.95)
            self.crossover_rate = min(1.0, self.crossover_rate + 0.05)

    def __call__(self, func):
        bounds = func.bounds
        self.population = bounds.lb + (bounds.ub - bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = self._evaluate(self.population[i], func)
            if self.fitness[i] < self.best_fitness:
                self.best_solution = self.population[i]
                self.best_fitness = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_solution

        while self.evaluations < self.budget:
            self._update_population(func, bounds)
            for i in range(self.population_size):
                fitness = self._evaluate(self.population[i], func)
                if fitness < self.fitness[i]:
                    self.fitness[i] = fitness
                    if fitness < self.best_fitness:
                        self.best_solution = self.population[i]
                        self.best_fitness = fitness
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return self.best_solution
            self._adapt_parameters()

        return self.best_solution