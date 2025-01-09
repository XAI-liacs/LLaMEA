import numpy as np

class AdaptiveParticleSwarm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w = 0.7  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        local_best_positions = np.copy(particles)
        local_best_scores = np.array([func(p) for p in particles])

        global_best_position = particles[np.argmin(local_best_scores)]
        global_best_score = np.min(local_best_scores)

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (local_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                current_score = func(particles[i])
                evaluations += 1

                if current_score < local_best_scores[i]:
                    local_best_positions[i] = particles[i]
                    local_best_scores[i] = current_score

                    if current_score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = current_score

                if evaluations >= self.budget:
                    break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt