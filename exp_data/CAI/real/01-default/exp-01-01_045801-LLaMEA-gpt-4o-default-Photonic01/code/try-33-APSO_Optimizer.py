import numpy as np

class APSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.inertia_weight = 0.9  # Initial inertia weight
        self.evaluations = 0
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([self.evaluate(ind) for ind in self.positions])
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def update_velocities_and_positions(self, lb, ub):
        r1, r2 = np.random.rand(), np.random.rand()
        self.inertia_weight = 0.4 + 0.5 * (1 - self.evaluations / self.budget)  # Dynamic inertia weight
        for i in range(self.population_size):
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = (
                self.inertia_weight * self.velocities[i] +
                cognitive_velocity +
                social_velocity
            )
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)

    def update_personal_best(self):
        for i in range(self.population_size):
            current_score = self.evaluate(self.positions[i])
            if current_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = current_score
                self.personal_best_positions[i] = self.positions[i].copy()
            if current_score < self.global_best_score:
                self.global_best_score = current_score
                self.global_best_position = self.positions[i].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.update_velocities_and_positions(lb, ub)
            self.update_personal_best()
            self.evaluations += self.population_size

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}