import numpy as np

class ImprovedAdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity_clamp = (ub - lb) * 0.1
        
        # Initialize positions, velocities, and neighbors
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-velocity_clamp, velocity_clamp, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(p) for p in positions])
        global_best_index = np.argmin(personal_best_values)
        global_best_position = positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        evaluations = self.num_particles

        while evaluations < self.budget:
            # Non-linear inertia reduction for better convergence
            inertia = self.inertia_min + (self.inertia_max - self.inertia_min) * np.exp(-3 * evaluations / self.budget)
            
            for i in range(self.num_particles):
                # Neighborhood best update
                neighbors = np.random.choice(self.num_particles, size=5, replace=False)
                neighborhood_best_position = personal_best_positions[neighbors[np.argmin(personal_best_values[neighbors])]]
                
                velocities[i] = (inertia * velocities[i] +
                                 self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * np.random.rand(self.dim) * (neighborhood_best_position - positions[i]))

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)

                # Update positions
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                # Evaluate fitness
                fitness = func(positions[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best_values[i] = fitness
                    personal_best_positions[i] = positions[i]

                # Update global best
                if fitness < global_best_value:
                    global_best_value = fitness
                    global_best_position = positions[i]

            # Early stopping if budget is exhausted
            if evaluations >= self.budget:
                break

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt