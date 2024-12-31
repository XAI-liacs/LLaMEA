import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        w_max, w_min = 0.9, 0.4
        c1, c2 = 2.0, 2.0
        v_max = 0.2 * (ub - lb)
        iteration = 0

        while iteration < self.budget // self.num_particles:
            w = w_max - ((w_max - w_min) * iteration / (self.budget // self.num_particles))
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i]))
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                f_value = func(particles[i])
                if f_value < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = f_value

                if f_value < global_best_value:
                    global_best = particles[i]
                    global_best_value = f_value

            iteration += 1

        self.f_opt = global_best_value
        self.x_opt = global_best
        return self.f_opt, self.x_opt