import numpy as np

class HybridAdaptiveQuantumBees:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 10 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_position = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.neighborhood_size = 0.1  # Percentage of dimension for local search
        self.alpha = 0.5  # Probability to use quantum adaptation
        self.beta = 0.5  # Adaptive parameter for local bee search

    def quantum_adaptation(self):
        scale = 0.1
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / 3))

    def local_bee_search(self, position, func):
        new_position = position + self.beta * (np.random.rand(self.dim) - 0.5) * 2 * self.neighborhood_size
        new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
        new_score = func(new_position)
        if new_score < self.best_score:
            self.best_position = new_position
            self.best_score = new_score
        return new_position, new_score

    def update_positions(self, func):
        for i in range(self.population_size):
            if np.random.rand() < self.alpha:
                new_position = self.positions[i] + self.quantum_adaptation()
            else:
                new_position, new_score = self.local_bee_search(self.positions[i], func)
                if new_score < self.fitness[i]:
                    self.positions[i] = new_position
                    self.fitness[i] = new_score
                    continue

            new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
            new_score = func(new_position)
            self.evaluations += 1

            if new_score < self.fitness[i]:
                self.positions[i] = new_position
                self.fitness[i] = new_score

            if new_score < self.best_score:
                self.best_position = new_position
                self.best_score = new_score

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.fitness[i] = score
            if score < self.best_score:
                self.best_position = self.positions[i]
                self.best_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_position

        while self.evaluations < self.budget:
            self.update_positions(func)
            self.beta = min(1.0, self.beta + 0.02)

        return self.best_position