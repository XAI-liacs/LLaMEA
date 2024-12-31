import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        # Initialize particles' positions and velocities
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.Inf)
        
        # Initialize global best
        global_best_value = np.Inf
        global_best_position = None

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate the function at current position
                current_value = func(positions[i])
                evaluations += 1
                
                # Update personal and global best
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = positions[i]
                
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = positions[i]

                # Early stopping condition
                if evaluations >= self.budget:
                    break

            # Adaptive inertia weight
            w = 0.4 + (0.5 * np.cos(np.pi * evaluations / self.budget))
            c1, c2 = 2.0, 2.0  # Cognitive and social coefficients

            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = c1 * r1 * (personal_best_positions[i] - positions[i])
                social_component = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt