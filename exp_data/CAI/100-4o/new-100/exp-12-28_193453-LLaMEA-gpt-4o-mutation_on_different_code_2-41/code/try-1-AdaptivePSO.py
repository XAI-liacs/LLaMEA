import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.w = 0.9  # initial inertia weight
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        v_max = 0.1 * (ub - lb)  # max velocity clamping

        # Initialize swarm
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-v_max, v_max, (self.swarm_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_values = np.full(self.swarm_size, np.Inf)

        for _ in range(self.budget // self.swarm_size):
            # Evaluate function
            for i in range(self.swarm_size):
                f_value = func(positions[i])
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i].copy()
                if f_value < self.f_opt:
                    self.f_opt = f_value
                    self.x_opt = positions[i].copy()

            # Update particles
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * ((self.budget - _ * self.swarm_size) / self.budget) * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (self.x_opt - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social

                # Apply velocity clamping
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

            # Update inertia weight
            self.w = 0.4 + 0.5 * ((self.budget - _ * self.swarm_size) / self.budget)

        return self.f_opt, self.x_opt