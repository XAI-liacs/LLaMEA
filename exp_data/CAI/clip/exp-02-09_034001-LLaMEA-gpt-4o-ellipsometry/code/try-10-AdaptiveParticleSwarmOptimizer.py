import numpy as np

class AdaptiveParticleSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 10 * self.dim  # Heuristic choice
        max_vel = 0.2 * (ub - lb)
        min_vel = -max_vel
        inertia_weight = 0.9  # Initial inertia weight
        cognitive_coeff = 2.0
        social_coeff = 2.0
        np.random.seed(42)

        # Initialize particle positions and velocities
        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(min_vel, max_vel, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(ind) for ind in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        evaluations = num_particles

        while evaluations < self.budget:
            for i in range(num_particles):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = social_coeff * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], min_vel, max_vel)

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate new fitness
                fitness = func(positions[i])
                evaluations += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i]

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

            # Dynamically update inertia weight
            inertia_weight = 0.4 + 0.5 * ((self.budget - evaluations) / self.budget)

        # Return the best found solution
        return global_best_position, global_best_fitness