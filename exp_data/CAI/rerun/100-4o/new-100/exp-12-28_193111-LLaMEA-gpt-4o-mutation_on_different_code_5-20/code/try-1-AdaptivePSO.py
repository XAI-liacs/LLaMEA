import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.w = 0.9  # inertia weight
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        for evals in range(0, self.budget, self.num_particles):
            for i in range(self.num_particles):
                if evals + i >= self.budget:
                    break
                
                score = func(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]

            global_best_position = particles[np.argmin(personal_best_scores)]
            self.w = 0.5 + 0.4 * (1 - (evals / self.budget))  # Improved dynamic inertia weight

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = self.c2 * r2 * (global_best_position - particles[i])
                velocities[i] = self.w * velocities[i] + cognitive_velocity + social_velocity

                # Modified adaptive velocity clamping
                velocities[i] = np.clip(velocities[i], -0.3 * (bounds[1] - bounds[0]), 0.3 * (bounds[1] - bounds[0]))

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], bounds[0], bounds[1])

        return self.f_opt, self.x_opt