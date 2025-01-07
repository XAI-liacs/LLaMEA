import numpy as np

class HybridQIGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 10 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.crossover_rate = 0.5
        self.mutation_factor = 0.8

    def _initialize_population(self, bounds):
        self.positions = bounds.lb + (bounds.ub - bounds.lb) * np.random.rand(self.population_size, self.dim)

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.fitness[i] = score
            if score < self.best_fitness:
                self.best_fitness = score
                self.best_solution = self.positions[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break

    def _mutation(self, idx):
        idxs = list(range(self.population_size))
        idxs.remove(idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        return np.clip(mutant, 0, 1)

    def _crossover(self, target, mutant):
        crossover = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        return crossover

    def _select(self, target_idx, candidate, func):
        candidate_fitness = func(candidate)
        if candidate_fitness < self.fitness[target_idx]:
            self.positions[target_idx] = candidate
            self.fitness[target_idx] = candidate_fitness
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate
                self.best_fitness = candidate_fitness
        self.evaluations += 1

    def __call__(self, func):
        self._initialize_population(func.bounds)
        self._evaluate_population(func)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutation(i)
                crossover = self._crossover(self.positions[i], mutant)
                crossover = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * crossover
                self._select(i, crossover, func)
                if self.evaluations >= self.budget:
                    break

        return self.best_solution