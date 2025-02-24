import numpy as np

class AdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.positions = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.729
        self.cognitive_coef = 1.49445
        self.social_coef = 1.49445

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * self.positions

        for step in range(self.budget):
            # Evaluate current positions
            for i in range(self.population_size):
                fitness = func(self.positions[i])
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()

            # Update global best
            min_personal_best_score = np.min(self.personal_best_scores)
            if min_personal_best_score < self.global_best_score:
                self.global_best_score = min_personal_best_score
                self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)].copy()

            # Adaptive adjustment of coefficients
            self.cognitive_coef = np.random.uniform(1.4, 2.0)
            self.social_coef = np.random.uniform(1.4, 2.0)

            # Introduce chaotic search behavior
            if step > self.budget // 2:
                chaos_factor = 0.5 * np.sin(step)
            else:
                chaos_factor = 0.1

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coef * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_coef * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.positions[i] += self.velocities[i] + chaos_factor * np.random.randn(self.dim)
                self.positions[i] = np.clip(self.positions[i], lb, ub)

        return self.global_best_position