import numpy as np

class OptimizedEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.inertia_weight = 0.729
        self.cognitive_const = 1.49445
        self.social_const = 1.49445
        self.particles = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.copy(self.particles[0])
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.particles)
            self.evaluations += self.population_size

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.particles[better_mask]
            
            min_score_index = np.argmin(scores)
            if scores[min_score_index] < self.global_best_score:
                self.global_best_score = scores[min_score_index]
                self.global_best_position = self.particles[min_score_index].copy()

            random_factors = np.random.rand(self.population_size, self.dim, 2)
            cognitive_component = self.cognitive_const * random_factors[:, :, 0] * (self.personal_best_positions - self.particles)
            social_component = self.social_const * random_factors[:, :, 1] * (self.global_best_position - self.particles)

            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.particles += self.velocities
            np.clip(self.particles, -5, 5, out=self.particles)

        return self.global_best_position, self.global_best_score