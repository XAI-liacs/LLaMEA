import numpy as np

class QCPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5  # Quantum cloud size
        self.population = None
        self.velocities = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.pbest = None
        self.pbest_scores = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.pbest = self.population.copy()
        self.pbest_scores = self.scores.copy()
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_best(self):
        for i in range(self.population_size):
            if self.scores[i] < self.pbest_scores[i]:
                self.pbest[i] = self.population[i].copy()
                self.pbest_scores[i] = self.scores[i]
        best_idx = np.argmin(self.pbest_scores)
        if self.pbest_scores[best_idx] < self.best_score:
            self.best_solution = self.pbest[best_idx].copy()
            self.best_score = self.pbest_scores[best_idx]

    def quantum_position_update(self, lb, ub):
        for i in range(self.population_size):
            # Cooperative term
            gbest = self.best_solution
            p = self.pbest[i]
            r1, r2 = np.random.rand(), np.random.rand()
            mbest = np.mean(self.pbest, axis=0)
            u = r1 * p + r2 * gbest + (1 - r1 - r2) * mbest
            phi = np.random.uniform(-1, 1, self.dim)
            self.population[i] = u + self.alpha * phi * np.abs(self.population[i] - u)
            self.population[i] = np.clip(self.population[i], lb, ub)
            self.scores[i] = self.evaluate(self.population[i])
            self.evaluations += 1

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.quantum_position_update(lb, ub)
            self.update_best()
            if self.evaluations >= self.budget:
                break

        return {'solution': self.best_solution, 'fitness': self.best_score}