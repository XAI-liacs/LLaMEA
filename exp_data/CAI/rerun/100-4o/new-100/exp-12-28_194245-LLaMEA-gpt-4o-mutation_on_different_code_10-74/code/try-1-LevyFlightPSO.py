import numpy as np

class LevyFlightPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 50
        self.w_max = 0.9  # initial inertia weight
        self.w_min = 0.4  # final inertia weight
        self.c1 = 1.5  # cognitive (particle) weight
        self.c2 = 1.5  # social (swarm) weight

    def levy_flight(self, L):
        beta = 1.5
        sigma_u = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=L)
        v = np.random.normal(0, 1, size=L)
        return u / np.abs(v) ** (1 / beta)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particles' positions and velocities
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, np.Inf)

        # Main optimization loop
        for t in range(self.budget // self.num_particles):
            self.w = self.w_max - (self.w_max - self.w_min) * (t / (self.budget // self.num_particles))
            for i, x in enumerate(positions):
                f = func(x)
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = x
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

            # Update velocities and positions
            g_best = personal_best_positions[np.argmin(personal_best_scores)]
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] * 0.9 +  # damping factor for stability
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (g_best - positions[i]) +
                                 self.levy_flight(self.dim) * 0.01)
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

        return self.f_opt, self.x_opt