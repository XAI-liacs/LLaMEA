import numpy as np

class AdaptiveHybridGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(15, 6 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_individual = None
        self.best_fitness = float('inf')
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.adaptive_rate = 0.2
        self.evaluations = 0

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_individual = self.population[i].copy()
            self.evaluations += 1

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim)
            child = np.concatenate([parent1[:point], parent2[point:]])
        else:
            child = parent1.copy()
        return child

    def _mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(self.dim)
            individual[idx] += np.random.normal(0, 0.1)
        return np.clip(individual, 0, 1)

    def _adaptive_differential_evolution(self, individual, best_individual):
        mutant = individual + self.adaptive_rate * (best_individual - individual)
        return np.clip(mutant, 0, 1)

    def _select_parents(self):
        idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
        return self.population[idx1], self.population[idx2]

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        self._evaluate_population(func)

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self._select_parents()
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                child = self._adaptive_differential_evolution(child, self.best_individual)
                new_population.append(child)
            self.population = np.array(new_population)
            self._evaluate_population(func)

        return self.best_individual