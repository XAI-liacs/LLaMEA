import numpy as np

class EnhancedPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 40
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.w_min = 0.4
        self.w_max = 0.9
        self.particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.num_particles, np.Inf)
        self.global_best_position = None

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.num_particles):
                f = func(self.particles[i])
                eval_count += 1
                if f < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = f
                    self.personal_best_positions[i] = self.particles[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = self.particles[i]
                    self.global_best_position = np.copy(self.x_opt)

            # Update velocities and positions
            w = self.w_max - ((self.w_max - self.w_min) * (eval_count / self.budget))
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    w * self.velocities[i] +
                    self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                    self.c2 * r2 * (self.global_best_position - self.particles[i])
                )
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)

        return self.f_opt, self.x_opt