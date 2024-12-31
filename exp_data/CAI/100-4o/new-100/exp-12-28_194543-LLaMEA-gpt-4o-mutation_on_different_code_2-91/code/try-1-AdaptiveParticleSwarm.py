import numpy as np

class AdaptiveParticleSwarm:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.c1 = 1.5  # Personal attraction coefficient
        self.c2 = 1.5  # Global attraction coefficient
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        
        # Initialize particles' positions and velocities
        positions = np.random.uniform(bounds[0], bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.Inf)

        global_best_position = None
        global_best_value = np.Inf
        
        eval_count = 0

        while eval_count < self.budget:
            # Adaptive inertia weight
            self.w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break
                
                # Evaluate the function at particle's position
                f_value = func(positions[i])
                eval_count += 1

                # Update personal best
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i]

                # Update global best
                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = positions[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                
                # Clamp velocity to the boundaries
                velocities[i] = np.clip(velocities[i], bounds[0] - positions[i], bounds[1] - positions[i])

                # Update position
                positions[i] += velocities[i]

                # Clamp position to the boundaries
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt