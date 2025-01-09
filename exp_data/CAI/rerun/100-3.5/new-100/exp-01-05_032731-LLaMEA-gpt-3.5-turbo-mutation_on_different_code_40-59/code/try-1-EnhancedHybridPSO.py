import numpy as np

class EnhancedHybridPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=20, omega=0.5, phi_p=0.8, phi_g=0.8, alpha=0.2):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.alpha = alpha
        self.f_opt = np.Inf
        self.x_opt = None

    def _local_search(self, x, func, step_size):
        perturbation = np.random.uniform(-step_size, step_size, size=self.dim)
        x_new = np.clip(x + perturbation, -5.0, 5.0)
        f_new = func(x_new)
        if f_new < func(x):
            return x_new
        return x

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        inertia_weights = np.linspace(self.omega, 0.1, self.budget) 

        for t in range(self.budget):
            for i in range(self.swarm_size):
                x = swarm[i]
                velocity = velocities[i]
                p_best = swarm[np.argmin([func(p) for p in swarm])]
                g_best = swarm[np.argmin([func(p) for p in swarm])]

                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                velocities[i] = inertia_weights[t] * velocity + self.phi_p * r_p * (p_best - x) + self.phi_g * r_g * (g_best - x)
                swarm[i] = np.clip(x + velocities[i], -5.0, 5.0)
                step_size = 0.5 * (1 - t / self.budget) * self.alpha
                swarm[i] = self._local_search(swarm[i], func, step_size)

                f = func(swarm[i])
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = swarm[i]

        return self.f_opt, self.x_opt