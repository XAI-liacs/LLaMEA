import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w_max = 0.9
        self.w_min = 0.4
        self.v_max = 0.5  # maximum velocity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.swarm_size, self.dim))
        personal_best = particles.copy()
        personal_best_f = np.full(self.swarm_size, np.Inf)
        global_best = particles[0]
        global_best_f = np.Inf

        for i in range(self.budget//self.swarm_size):
            for j in range(self.swarm_size):
                f = func(particles[j])
                if f < personal_best_f[j]:
                    personal_best_f[j] = f
                    personal_best[j] = particles[j].copy()

                if f < global_best_f:
                    global_best_f = f
                    global_best = particles[j].copy()

            w = self.w_max - ((self.w_max - self.w_min) * i / (self.budget//self.swarm_size))
            r1, r2 = np.random.rand(2)
            adaptive_vmax = self.v_max * (global_best_f / (global_best_f + 1))  # Adaptive v_max
            for j in range(self.swarm_size):
                velocities[j] = (
                    w * velocities[j]
                    + self.c1 * r1 * (personal_best[j] - particles[j])
                    + self.c2 * r2 * (global_best - particles[j])
                )
                velocities[j] = np.clip(velocities[j], -adaptive_vmax, adaptive_vmax)  # Adaptive clamping
                particles[j] += velocities[j]
                particles[j] = np.clip(particles[j], lb, ub)

        self.f_opt, self.x_opt = global_best_f, global_best
        return self.f_opt, self.x_opt