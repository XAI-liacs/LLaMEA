import numpy as np

class PSOAdaptive:
    def __init__(self, budget=10000, dim=10, num_particles=30, omega_min=0.4, omega_max=0.9, phi_p=0.5, phi_g=0.5, ls_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.ls_rate = ls_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def local_search(x):
            new_x = x + np.random.uniform(-self.ls_rate, self.ls_rate, size=self.dim)
            return new_x

        def evaluate_particle(x):
            return func(x)

        swarm = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        p_best = swarm.copy()
        p_best_vals = np.array([func(x) for x in p_best])
        g_best_idx = np.argmin(p_best_vals)
        g_best = p_best[g_best_idx].copy()
        omega = self.omega_max

        for _ in range(self.budget):
            for i in range(self.num_particles):
                r_p = np.random.uniform(0, 1, self.dim)
                r_g = np.random.uniform(0, 1, self.dim)
                velocities[i] = omega * velocities[i] + self.phi_p * r_p * (p_best[i] - swarm[i]) + self.phi_g * r_g * (g_best - swarm[i])
                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)

                swarm[i] = local_search(swarm[i])
                f_val = evaluate_particle(swarm[i])

                if f_val < p_best_vals[i]:
                    p_best[i] = swarm[i].copy()
                    p_best_vals[i] = f_val

                    if f_val < self.f_opt:
                        self.f_opt = f_val
                        self.x_opt = swarm[i].copy()
                        g_best = swarm[i].copy()

            # Update inertia weight based on fitness improvement
            current_best = min(p_best_vals)
            improvement = self.f_opt - current_best
            if improvement > 0:
                omega = max(self.omega_min, min(self.omega_max, omega + 0.1 * improvement))

        return self.f_opt, self.x_opt