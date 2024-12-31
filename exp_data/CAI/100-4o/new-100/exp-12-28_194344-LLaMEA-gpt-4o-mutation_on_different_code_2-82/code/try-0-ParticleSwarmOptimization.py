import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, c1=1.5, c2=2.0, inertia=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1  # Cognitive component
        self.c2 = c2  # Social component
        self.inertia = inertia  # Inertia weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize swarm's positions and velocities
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(pos) for pos in positions])
        
        # Initialize global best
        global_best_value = np.min(personal_best_values)
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate current position
                current_value = func(positions[i])
                evaluations += 1

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = positions[i]

                # Update global best
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = positions[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = self.inertia * velocities[i] + cognitive_component + social_component
                
                # Update position with bounds check
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

            # Check budget constraint
            if evaluations >= self.budget:
                break

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt