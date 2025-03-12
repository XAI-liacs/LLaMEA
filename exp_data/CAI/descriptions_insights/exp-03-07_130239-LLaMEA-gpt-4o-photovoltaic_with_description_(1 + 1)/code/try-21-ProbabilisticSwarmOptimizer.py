import numpy as np

class ProbabilisticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(0.1 * dim))
        self.position = None
        self.velocity = None
        self.best_position = None
        self.best_score = np.inf
        self.inertia_weight = 0.9
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.global_best_position = None
        self.momentum_factor = 0.1  # New: Momentum factor
        self.adaptive_learning_rate = 0.5  # New: Adaptive learning rate

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.best_position = self.position.copy()
        self.best_scores = np.full(self.population_size, np.inf)

    def _update_velocity_position(self):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_term = self.cognitive_weight * r1 * (self.best_position - self.position)
        social_term = self.social_weight * r2 * (self.global_best_position - self.position)
        self.velocity = self.inertia_weight * self.velocity + cognitive_term + social_term
        self.position += self.velocity + self.momentum_factor * np.sign(self.velocity)  # Modified: Added momentum factor
        self.position += np.random.normal(0, self.adaptive_learning_rate * (1 - self.inertia_weight), self.position.shape)  # Modified: Adaptive learning rate
        self.inertia_weight = max(0.4, self.inertia_weight * 0.99)  # Dynamic inertia adaptation
        self.momentum_factor = max(0.05, self.momentum_factor * 0.99)  # Dynamically adjust momentum factor

    def _evaluate_population(self, func):
        scores = np.apply_along_axis(func, 1, self.position)
        for i in range(self.population_size):
            if scores[i] < self.best_scores[i]:
                self.best_scores[i] = scores[i]
                self.best_position[i] = self.position[i].copy()
        current_best_score = np.min(scores)
        if current_best_score < self.best_score:
            self.best_score = current_best_score
            self.global_best_position = self.best_position[np.argmin(scores)].copy()

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0
        while evaluations < self.budget:
            self._evaluate_population(func)
            self._update_velocity_position()
            evaluations += self.population_size
            # Newly added line for dynamic weights adjustment
            self.cognitive_weight, self.social_weight = 1.5 - (0.1 * evaluations / self.budget), 1.5 + (0.1 * evaluations / self.budget)
        return self.global_best_position, self.best_score