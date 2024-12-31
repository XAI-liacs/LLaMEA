import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w_max = 0.9  # max inertia weight
        self.w_min = 0.4  # min inertia weight

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.Inf)

        # Evaluate initial positions
        for i in range(self.swarm_size):
            f = func(positions[i])
            if f < personal_best_values[i]:
                personal_best_values[i] = f
                personal_best_positions[i] = positions[i]
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = positions[i]

        # Budget counter
        evaluations = self.swarm_size

        while evaluations < self.budget:
            # Update inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))

            for i in range(self.swarm_size):
                # Update velocity
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * np.random.rand() * (self.x_opt - positions[i]))

                # Update position
                positions[i] = positions[i] + velocities[i]
                # Ensure particles are within bounds
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

                # Evaluate new position
                f = func(positions[i])
                evaluations += 1

                if f < personal_best_values[i]:
                    personal_best_values[i] = f
                    personal_best_positions[i] = positions[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = positions[i]

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt