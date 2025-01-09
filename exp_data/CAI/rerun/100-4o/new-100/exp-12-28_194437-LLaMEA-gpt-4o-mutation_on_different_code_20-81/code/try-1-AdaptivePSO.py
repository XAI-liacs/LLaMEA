import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = min(100, budget // 10)
        self.inertia_weight = 0.9  # Start with a higher inertia weight
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.velocity_clamp = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim)) * self.velocity_clamp
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        for iteration in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                score = func(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]

            # Update global best to local neighborhood best
            local_best_positions = personal_best_positions.copy()
            for i in range(self.num_particles):
                local_neighbors = personal_best_positions[max(0, i-2):min(self.num_particles, i+3)]
                local_best_positions[i] = local_neighbors[np.argmin([func(x) for x in local_neighbors])] 

            # Update inertia weight dynamically
            self.inertia_weight = 0.9 - (0.5 * (iteration / (self.budget // self.num_particles)))

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_weight * r1 * (personal_best_positions[i] - particles[i])
                    + self.social_weight * r2 * (local_best_positions[i] - particles[i])
                )
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        return self.f_opt, self.x_opt