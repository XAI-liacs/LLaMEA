import numpy as np

class QIPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.1  # Learning rate for personal best
        self.beta = 0.3   # Learning rate for global best
        self.evaluations = 0
        self.population = None
        self.velocities = None
        self.personal_bests = None
        self.personal_best_scores = None
        self.global_best = None
        self.global_best_score = float('inf')

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(lb, ub, (self.population_size, self.dim)) * 0.1
        self.personal_bests = self.population.copy()
        self.personal_best_scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best = self.personal_bests[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def quantum_update(self, lb, ub):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.alpha * r1 * (self.personal_bests[i] - self.population[i])
            social_component = self.beta * r2 * (self.global_best - self.population[i])
            self.velocities[i] = cognitive_component + social_component

            # Add a quantum-inspired exploration
            quantum_exploration = np.random.uniform(lb, ub, self.dim) * np.random.choice([-1, 1], self.dim)
            self.population[i] += self.velocities[i] + quantum_exploration
            self.population[i] = np.clip(self.population[i], lb, ub)

            # Evaluate and update personal and global bests
            score = self.evaluate(self.population[i])
            if score < self.personal_best_scores[i]:
                self.personal_bests[i] = self.population[i].copy()
                self.personal_best_scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.population[i].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.quantum_update(lb, ub)
            self.evaluations += self.population_size
            if self.evaluations >= self.budget:
                break

        return {'solution': self.global_best, 'fitness': self.global_best_score}