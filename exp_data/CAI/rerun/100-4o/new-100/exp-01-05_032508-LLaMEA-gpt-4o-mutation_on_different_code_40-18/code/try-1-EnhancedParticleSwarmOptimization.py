import numpy as np

class EnhancedParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 50
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 1.5
        self.c2 = 2.0
        self.v_max = 0.2 * (5.0 - (-5.0))
        self.v_min = -self.v_max

    def __call__(self, func):
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, np.Inf)
        global_best_position = None
        global_best_value = np.Inf

        evals = 0
        while evals < self.budget:
            for i in range(self.num_particles):
                f_val = func(particles[i])
                evals += 1
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best_positions[i] = particles[i].copy()
                if f_val < global_best_value:
                    global_best_value = f_val
                    global_best_position = particles[i].copy()

            if evals >= self.budget:
                break

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            c1 = self.c1_max - ((self.c1_max - self.c1_min) * (evals / self.budget))

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * r2 * (global_best_position - particles[i])
                velocities[i] = (
                    inertia_weight * velocities[i] +
                    cognitive_component +
                    social_component
                )
                
                velocities[i] = np.tanh(velocities[i]) * self.v_max

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt