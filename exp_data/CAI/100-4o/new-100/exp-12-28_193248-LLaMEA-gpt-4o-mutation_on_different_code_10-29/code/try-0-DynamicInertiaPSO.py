import numpy as np

class DynamicInertiaPSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        # Evaluate initial positions
        for i in range(self.num_particles):
            f = func(positions[i])
            personal_best_scores[i] = f
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = positions[i]

        # PSO parameters
        inertia_weight_max = 0.9
        inertia_weight_min = 0.4
        c1 = 2.0
        c2 = 2.0
        iteration = 0
        evaluations = self.num_particles

        while evaluations < self.budget:
            # Dynamic inertia weight
            inertia_weight = (inertia_weight_max - inertia_weight_min) * (1 - (evaluations / self.budget)) + inertia_weight_min
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                # Update velocity
                velocities[i] = (inertia_weight * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 c2 * r2 * (self.x_opt - positions[i]))
                # Update position
                positions[i] += velocities[i]
                # Clip positions to bounds
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                # Evaluate new position
                f = func(positions[i])
                evaluations += 1
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = positions[i]

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = positions[i]

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt