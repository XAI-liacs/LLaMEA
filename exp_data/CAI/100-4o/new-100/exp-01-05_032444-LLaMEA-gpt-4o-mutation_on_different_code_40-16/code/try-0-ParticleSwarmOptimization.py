import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive (personal) coefficient
        self.c2 = c2  # social (global) coefficient
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, np.inf)
        global_best_value = np.inf
        global_best_position = None

        for _ in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                f_value = func(particles[i])
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = particles[i]

                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = particles[i]

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        self.f_opt, self.x_opt = global_best_value, global_best_position
        return self.f_opt, self.x_opt