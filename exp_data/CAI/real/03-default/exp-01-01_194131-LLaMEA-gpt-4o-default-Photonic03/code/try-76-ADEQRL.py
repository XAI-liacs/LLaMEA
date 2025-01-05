import numpy as np

class ADEQRL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_idx = None
        self.best_fitness = float('inf')
        self.f_cr = 0.9
        self.f_f = 0.8
        self.evaluations = 0
        self.quantum_rotation = 0.1

    def _quantum_rotation(self, current):
        theta = np.random.uniform(-np.pi, np.pi, self.dim)
        rotation_matrix = np.cos(theta) * current + np.sin(theta) * (1 - current)
        return rotation_matrix

    def _mutate(self, a, b, c):
        mutant = np.clip(a + self.f_f * (b - c), 0, 1)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.f_cr
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def _select(self, idx, candidate, func):
        candidate_real = func.bounds.lb + candidate * (func.bounds.ub - func.bounds.lb)
        candidate_fitness = func(candidate_real)
        if candidate_fitness < self.fitness[idx]:
            self.positions[idx] = candidate
            self.fitness[idx] = candidate_fitness
            if candidate_fitness < self.best_fitness:
                self.best_idx = idx
                self.best_fitness = candidate_fitness
        self.evaluations += 1

    def __call__(self, func):
        self.positions = np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            real_pos = func.bounds.lb + self.positions[i] * (func.bounds.ub - func.bounds.lb)
            self.fitness[i] = func(real_pos)
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_idx = i
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return func.bounds.lb + self.positions[self.best_idx] * (func.bounds.ub - func.bounds.lb)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self._mutate(self.positions[a], self.positions[b], self.positions[c])
                trial = self._crossover(self.positions[i], mutant)
                trial = np.clip(trial + self.quantum_rotation * self._quantum_rotation(trial), 0, 1)
                self._select(i, trial, func)
                if self.evaluations >= self.budget:
                    break

        return func.bounds.lb + self.positions[self.best_idx] * (func.bounds.ub - func.bounds.lb)