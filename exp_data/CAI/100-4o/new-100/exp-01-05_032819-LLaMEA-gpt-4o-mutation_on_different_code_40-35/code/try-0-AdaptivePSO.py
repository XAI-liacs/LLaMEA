import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        evals = 0
        while evals < self.budget:
            for i in range(self.num_particles):
                if evals >= self.budget:
                    break
                score = func(particles[i])
                evals += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]

            w = self.w_max - (evals / self.budget) * (self.w_max - self.w_min)
            for i in range(self.num_particles):
                inertia = w * velocities[i]
                cognitive = self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - particles[i])
                social = self.c2 * np.random.rand(self.dim) * (self.x_opt - particles[i])
                velocities[i] = inertia + cognitive + social

                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        return self.f_opt, self.x_opt