import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        w = 0.5  # inertia weight
        c1 = 1.5  # cognitive parameter
        c2 = 1.5  # social parameter

        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.inf)

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Evaluate current particle
                score = func(particles[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                # Update global best
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]

                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best_positions[i] - particles[i])
                    + c2 * r2 * (self.x_opt - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

        return self.f_opt, self.x_opt