import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.05
        self.c2 = 2.05
        self.x_opt = None
        self.f_opt = np.Inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, np.Inf)
        global_best_value = np.Inf
        global_best_position = None

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                current_value = func(particles[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particles[i]

                if evaluations >= self.budget:
                    break

            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * r2 * (global_best_position - particles[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        self.x_opt = global_best_position
        self.f_opt = global_best_value
        return self.f_opt, self.x_opt