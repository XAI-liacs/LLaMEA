import numpy as np

class EnhancedPSO:
    def __init__(self, budget=10000, dim=10, num_particles=30, w_max=0.9, w_min=0.4, c1=2.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.inf)

        global_best_score = np.inf
        global_best_position = None

        for i in range(self.budget):
            if i % 100 == 0:  # Diversification step
                self._diversify_particles(particles, lb, ub, 0.1)

            idx = i % self.num_particles
            fitness = func(particles[idx])

            if fitness < personal_best_scores[idx]:
                personal_best_scores[idx] = fitness
                personal_best_positions[idx] = particles[idx]

            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = particles[idx]

            w = self.w_max - i * (self.w_max - self.w_min) / self.budget

            r1, r2 = np.random.rand(), np.random.rand()
            velocities[idx] = (w * velocities[idx] +
                               self.c1 * r1 * (personal_best_positions[idx] - particles[idx]) +
                               self.c2 * r2 * (global_best_position - particles[idx]))
            particles[idx] = np.clip(particles[idx] + velocities[idx], lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt

    def _diversify_particles(self, particles, lb, ub, zone):
        for i in range(self.num_particles):
            if np.random.rand() < zone:
                particles[i] = np.random.uniform(lb, ub)