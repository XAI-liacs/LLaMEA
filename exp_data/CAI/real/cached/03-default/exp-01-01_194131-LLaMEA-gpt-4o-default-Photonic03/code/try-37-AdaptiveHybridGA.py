import numpy as np

class AdaptiveHybridGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 10 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.1
        self.evaluations = 0

    def quantum_mutation(self, individual):
        scale = 0.05
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return individual + step

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def select_parents(self):
        idx = np.random.choice(np.arange(self.population_size), size=2, replace=False, p=self.fitness / self.fitness.sum())
        return self.population[idx[0]], self.population[idx[1]]

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            fitness = func(self.population[i])
            self.fitness[i] = fitness
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.population[i].copy()
            self.evaluations += 1

    def _update_population(self, func):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select_parents()
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            new_population.append(child1)
            new_population.append(child2)
        if self.evaluations + len(new_population) >= self.budget:
            new_population = new_population[:self.budget - self.evaluations]

        for i in range(len(new_population)):
            if np.random.rand() < self.mutation_rate:
                new_population[i] = self.quantum_mutation(new_population[i])
            new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

        self.population = np.array(new_population)
        self.fitness = np.full(len(self.population), float('inf'))
        self._evaluate_population(func)

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        self._evaluate_population(func)

        while self.evaluations < self.budget:
            self._update_population(func)
            if self.evaluations % (self.budget // 4) == 0:
                self.mutation_rate = min(0.5, self.mutation_rate + 0.05)

        return self.best_solution