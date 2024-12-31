import numpy as np

class PSO:
    def __init__(self, budget=10000, dim=10, num_particles=30, c1=2.0, c2=2.0, w=0.7):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.f_opt = np.Inf
        self.x_opt = None
        self.velocities = np.zeros((num_particles, dim))
        self.particles = np.random.uniform(-5.0, 5.0, (num_particles, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(num_particles, np.Inf)
        self.global_best_value = np.Inf
        self.global_best_position = None

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            for i in range(self.num_particles):
                if evals >= self.budget:
                    break
                f = func(self.particles[i])
                evals += 1
                if f < self.personal_best_values[i]:
                    self.personal_best_values[i] = f
                    self.personal_best_positions[i] = self.particles[i]
                if f < self.global_best_value:
                    self.global_best_value = f
                    self.global_best_position = self.particles[i]

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i]))
                self.particles[i] = self.particles[i] + self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)
            self.w = 0.9 - (0.5 * evals / self.budget)  # Adaptive inertia weight

        return self.global_best_value, self.global_best_position