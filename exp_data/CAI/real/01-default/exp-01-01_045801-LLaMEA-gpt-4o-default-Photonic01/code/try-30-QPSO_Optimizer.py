import numpy as np

class QPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.alpha = 0.5  # Contraction-expansion coefficient
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.array([self.evaluate(pos) for pos in self.positions])
        best_idx = np.argmin(self.personal_best_fitness)
        self.best_global_position = self.personal_best_positions[best_idx]
        self.best_global_fitness = self.personal_best_fitness[best_idx]

    def evaluate(self, position):
        return self.func(position)

    def update_particles(self, lb, ub):
        for i in range(self.population_size):
            p_best = self.personal_best_positions[i]
            g_best = self.best_global_position
            u = np.random.uniform(0, 1, self.dim)
            mean_best = self.alpha * p_best + (1 - self.alpha) * g_best
            self.positions[i] = mean_best + self.alpha * np.log(1 / u) * np.sign(u - 0.5)
            self.positions[i] = np.clip(self.positions[i], lb, ub)
            fitness = self.evaluate(self.positions[i])
            self.evaluations += 1
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_positions[i] = self.positions[i]
                self.personal_best_fitness[i] = fitness
            if fitness < self.best_global_fitness:
                self.best_global_position = self.positions[i]
                self.best_global_fitness = fitness

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.update_particles(lb, ub)
            if self.evaluations >= self.budget:
                break

        return {'solution': self.best_global_position, 'fitness': self.best_global_fitness}