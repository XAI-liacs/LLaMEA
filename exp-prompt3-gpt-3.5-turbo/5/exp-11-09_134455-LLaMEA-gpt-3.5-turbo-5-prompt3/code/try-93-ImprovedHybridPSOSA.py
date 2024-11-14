class ImprovedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(10, int(20 * (1 - min(1, budget / 10000))))
        self.max_iter = budget // self.num_particles
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min = 0.4
        self.w_max = 0.9
        self.T_init = 1000.0
        self.T_min = 1e-8
        self.diversity_factor = 0.1
        self.dynamic_inertia = 0.5  # New parameter for dynamic inertia weight

    def __call__(self, func):
        def pso_sa_optimize():
            positions = np.zeros((self.num_particles, self.dim))
            for d in range(self.dim):
                step_size = 10.0 / self.num_particles
                for p in range(self.num_particles):
                    positions[p][d] = np.random.uniform(step_size * p, step_size * (p + 1))

            velocities = np.zeros((self.num_particles, self.dim))
            pbest_positions = np.copy(positions)
            pbest_values = np.array([func(p) for p in pbest_positions])
            gbest_position = pbest_positions[np.argmin(pbest_values)]
            gbest_value = np.min(pbest_values)
            T = self.T_init

            for _ in range(self.max_iter):
                for i in range(self.num_particles):
                    w = self.w_min + self.dynamic_inertia * (_ / self.max_iter) * (self.w_max - self.w_min)
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = w * velocities[i] + self.c1 * r1 * (pbest_positions[i] - positions[i]) + self.c2 * r2 * (gbest_position - positions[i])
                    positions[i] = np.clip(positions[i] + velocities[i], -5.0, 5.0)

                    candidate_position = positions[i] + np.random.normal(0, 0.1, size=self.dim)
                    candidate_position = np.clip(candidate_position, -5.0, 5.0)
                    candidate_value = func(candidate_position)

                    if candidate_value < pbest_values[i]:
                        pbest_positions[i] = candidate_position
                        pbest_values[i] = candidate_value

                    if candidate_value < gbest_value:
                        gbest_position = candidate_position
                        gbest_value = candidate_value
                    else:
                        delta = candidate_value - pbest_values[i]
                        if np.exp(-delta / T) > np.random.rand():
                            positions[i] = candidate_position
                            pbest_values[i] = candidate_value

                    res = minimize(func, positions[i], method='Nelder-Mead')
                    if res.fun < pbest_values[i]:
                        pbest_positions[i] = res.x
                        pbest_values[i] = res.fun

                    if res.fun < gbest_value:
                        gbest_position = res.x
                        gbest_value = res.fun

                T *= 0.99 if T > self.T_min else 1.0
                
                self.dynamic_inertia = 0.5 + 0.4 * (np.cos(0.9 * np.pi * _ / self.max_iter) + 1)  # Dynamic inertia update

                chaos_factor = 0.1
                for j in range(self.num_particles):
                    chaos_position = np.clip(positions[j] + chaos_factor * np.sin(positions[j] * np.pi), -5.0, 5.0)
                    chaos_value = func(chaos_position)
                    if chaos_value < pbest_values[j]:
                        positions[j] = chaos_position
                        pbest_positions[j] = chaos_position
                        pbest_values[j] = chaos_value

            return gbest_value

        return pso_sa_optimize()