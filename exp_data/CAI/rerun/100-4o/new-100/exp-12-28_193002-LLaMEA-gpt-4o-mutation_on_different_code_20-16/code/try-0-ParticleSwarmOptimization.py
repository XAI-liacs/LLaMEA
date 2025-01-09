import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize the swarm
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.Inf)
        
        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                f_value = func(positions[i])
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i]
                if f_value < self.f_opt:
                    self.f_opt = f_value
                    self.x_opt = positions[i]

            # Update velocities and positions
            global_best_position = personal_best_positions[np.argmin(personal_best_values)]
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.swarm_size):
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                    + self.c2 * r2 * (global_best_position - positions[i])
                )
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

        return self.f_opt, self.x_opt