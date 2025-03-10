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
        self.momentum_factor = 0.1
        self.adaptive_learning_rate = 0.5
        self.neighborhood_size = max(2, int(0.1 * self.population_size))  # New dynamic neighborhood size

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
        random_perturbation_scale = np.exp(-0.005 * self.budget)  # Updated line
        random_perturbation = np.random.normal(0, 0.1 * random_perturbation_scale, self.velocity.shape)
        self.velocity = self.inertia_weight * self.velocity + cognitive_term + social_term + random_perturbation
        self.position += self.velocity + self.momentum_factor * np.sign(self.velocity)
        self.position += np.random.normal(0, self.adaptive_learning_rate * (1 - self.inertia_weight), self.position.shape)
        self.inertia_weight = max(0.4, self.inertia_weight * (0.99 - 0.1 * (1 - self.inertia_weight)))  # Adaptive inertia decay
        self.momentum_factor = max(0.05, self.momentum_factor * 0.99)

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
            max_weight_change = 0.3
            temp_factor = np.exp(-evaluations / (0.1 * self.budget))
            self.cognitive_weight = 1.5 - (0.1 * evaluations / self.budget) * temp_factor
            self.social_weight = 1.5 + max_weight_change * (1 - temp_factor)
            self.adaptive_learning_rate = 0.5 * (1 + np.tanh(0.01 * (1 - self.best_score)))  # Adjust learning rate
            self.neighborhood_size = max(2, int(0.1 * self.population_size) + int(0.05 * (1 - self.best_score)))  # Adaptive neighborhood size
        return self.global_best_position, self.best_score