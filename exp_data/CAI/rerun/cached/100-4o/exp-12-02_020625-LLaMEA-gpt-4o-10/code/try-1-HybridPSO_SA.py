import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 10
        self.w = 0.7  # Inertia weight (increased for more exploration early on)
        self.c1 = 1.5  # Cognitive (particle) weight
        self.c2 = 1.7  # Social (swarm) weight (increased for better convergence)
        self.temp_start = 1.0  # Initial temperature for SA

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(pos) for pos in positions])
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        evals = self.num_particles
        while evals < self.budget:
            # Update velocities and positions for PSO
            r1, r2 = np.random.rand(2, self.num_particles, self.dim)
            self.w = 0.7 * (1 - evals / self.budget) + 0.3  # Adjust inertia weight over time
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            # Evaluate new solutions
            current_values = np.array([func(pos) for pos in positions])
            evals += self.num_particles

            # Update personal bests
            better_mask = current_values < personal_best_values
            personal_best_positions[better_mask] = positions[better_mask]
            personal_best_values[better_mask] = current_values[better_mask]

            # Update global best
            new_global_best_index = np.argmin(personal_best_values)
            new_global_best_value = personal_best_values[new_global_best_index]
            if new_global_best_value < global_best_value:
                global_best_position = personal_best_positions[new_global_best_index]
                global_best_value = new_global_best_value

            # Apply simulated annealing to refine the global best position
            current_temperature = self.temp_start * (1 - evals / self.budget)
            new_global_position = global_best_position + np.random.normal(0, 0.1, self.dim)
            new_global_position = np.clip(new_global_position, self.lower_bound, self.upper_bound)
            new_global_value = func(new_global_position)
            evals += 1
            if new_global_value < global_best_value or np.random.rand() < np.exp((global_best_value - new_global_value) / current_temperature):
                global_best_position = new_global_position
                global_best_value = new_global_value

        return global_best_value, global_best_position