import numpy as np

class QAGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.mutation_rate = 0.1
        self.selection_pressure = 1.5

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            score = func(self.positions[i])
            self.fitness[i] = score
            if score < self.best_fitness:
                self.best_fitness = score
                self.best_solution = self.positions[i].copy()
            self.evaluations += 1

    def _select_parents(self):
        fitness_inv = 1.0 / (self.fitness + 1e-12)
        prob = fitness_inv ** self.selection_pressure
        prob /= np.sum(prob)
        parents_idx = np.random.choice(self.population_size, size=self.population_size, p=prob)
        return self.positions[parents_idx]

    def _crossover(self, parents):
        offspring = parents.copy()
        for i in range(0, self.population_size, 2):
            if i+1 < self.population_size:
                alpha = np.random.rand(self.dim)
                offspring[i] = alpha * parents[i] + (1 - alpha) * parents[i+1]
                offspring[i+1] = alpha * parents[i+1] + (1 - alpha) * parents[i]
        return offspring

    def _mutate(self, offspring):
        mutation_matrix = np.random.rand(self.population_size, self.dim) < self.mutation_rate
        mutations = np.random.randn(self.population_size, self.dim) * mutation_matrix
        offspring += mutations
        return offspring

    def _adapt_mutation_rate(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.mutation_rate = min(0.5, self.mutation_rate + 0.01)

    def _adapt_selection_pressure(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.selection_pressure = min(2.5, self.selection_pressure + 0.1)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        self._evaluate_population(func)

        while self.evaluations < self.budget:
            parents = self._select_parents()
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)
            self.positions = np.clip(offspring, func.bounds.lb, func.bounds.ub)
            self._evaluate_population(func)
            self._adapt_mutation_rate()
            self._adapt_selection_pressure()

        return self.best_solution