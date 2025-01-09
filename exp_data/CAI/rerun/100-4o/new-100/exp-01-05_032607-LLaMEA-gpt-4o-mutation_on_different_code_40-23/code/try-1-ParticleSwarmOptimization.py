import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        w_max, w_min = 0.9, 0.4  # adaptive inertia weights
        c1 = 2.05  # cognitive parameter
        c2 = 2.05  # social parameter
        phi = c1 + c2
        kappa = 2 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi))  # constriction factor

        lb, ub = func.bounds.lb, func.bounds.ub

        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.inf)

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]

                if evaluations >= self.budget:
                    break

            w = w_max - ((w_max - w_min) * (evaluations / self.budget))

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    kappa * (w * velocities[i]
                    + c1 * r1 * (personal_best_positions[i] - particles[i])
                    + c2 * r2 * (self.x_opt - particles[i]))
                )
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

        return self.f_opt, self.x_opt