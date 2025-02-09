import numpy as np

class AdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25  # Increased population size
        self.positions = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.729  # Inertia weight to control exploration
        self.cognitive_coef = 1.49445
        self.social_coef = 1.49445

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * self.positions

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = func(self.positions[i])
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()

            min_personal_best_score = np.min(self.personal_best_scores)
            if min_personal_best_score < self.global_best_score:
                self.global_best_score = min_personal_best_score
                self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)].copy()

            self.cognitive_coef = np.random.uniform(1.3, 1.8)  # Adjusted range
            self.social_coef = np.random.uniform(1.3, 1.8)  # Adjusted range

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coef * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_coef * r2 * (self.global_best_position - self.positions[i])
                
                # Adaptive velocity scaling
                if np.random.rand() < 0.1:
                    self.velocities[i] *= np.random.uniform(0.5, 1.5)
                
                scaling_factor = 0.9  # Scaling factor added for velocity components
                self.velocities[i] = self.inertia_weight * self.velocities[i] + scaling_factor * (cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]
                # Diversity-enhancing perturbation
                if np.random.rand() < 0.05:
                    self.positions[i] += np.random.normal(0, 0.1, self.dim)
                self.positions[i] = np.clip(self.positions[i], lb, ub)

        return self.global_best_position