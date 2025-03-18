import numpy as np

class DynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def update_particle(self, i, func):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        self.velocities[i] = (
            self.inertia_weight * self.velocities[i] +
            self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.particles[i]) +
            self.social_coeff * r2 * (self.global_best_position - self.particles[i])
        )
        self.velocities[i] = np.clip(self.velocities[i], self.min_velocity, self.max_velocity)
        self.particles[i] += self.velocities[i]
        self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)

    def evaluate_particle(self, i, func):
        score = func(self.particles[i])
        self.evaluations += 1
        if score < self.personal_best_scores[i]:
            self.personal_best_scores[i] = score
            self.personal_best_positions[i] = self.particles[i]
        if score < self.global_best_score:
            self.global_best_score = score
            self.global_best_position = self.particles[i]

    def __call__(self, func):
        self.max_velocity = (func.bounds.ub - func.bounds.lb) * 0.2
        self.min_velocity = -(func.bounds.ub - func.bounds.lb) * 0.2
        self.particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(self.min_velocity, self.max_velocity, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations < self.budget:
                    self.update_particle(i, func)
                    self.evaluate_particle(i, func)
                    self.inertia_weight = 0.4 + 0.5 * (self.budget - self.evaluations) / self.budget

        return self.global_best_position, self.global_best_score