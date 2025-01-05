import numpy as np

class DALPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1_base = 2.0  # Cognitive coefficient base
        self.c2_base = 2.0  # Social coefficient base
        self.inertia = 0.5  # Inertia weight
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = self.population.copy()
        self.update_personal_best_scores()

    def update_personal_best_scores(self):
        for i in range(self.population_size):
            score = self.evaluate(self.population[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.population[i].copy()
            self.evaluations += 1

    def evaluate(self, solution):
        return self.func(solution)

    def update_velocity(self, idx, lb, ub):
        cognitive_component = np.random.rand(self.dim) * self.c1 * (self.personal_best_positions[idx] - self.population[idx])
        social_component = np.random.rand(self.dim) * self.c2 * (self.global_best_position - self.population[idx])
        self.velocities[idx] = self.inertia * self.velocities[idx] + cognitive_component + social_component
        self.velocities[idx] = np.clip(self.velocities[idx], -abs(ub-lb), abs(ub-lb))

    def update_position(self, idx, lb, ub):
        self.population[idx] += self.velocities[idx]
        self.population[idx] = np.clip(self.population[idx], lb, ub)

    def adapt_learning_coefficients(self):
        # Adapt coefficients based on the current evaluation ratio
        eval_ratio = self.evaluations / self.budget
        self.c1 = self.c1_base * (1 - eval_ratio)
        self.c2 = self.c2_base * eval_ratio

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.adapt_learning_coefficients()
            for i in range(self.population_size):
                self.update_velocity(i, lb, ub)
                self.update_position(i, lb, ub)
            self.update_personal_best_scores()

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}