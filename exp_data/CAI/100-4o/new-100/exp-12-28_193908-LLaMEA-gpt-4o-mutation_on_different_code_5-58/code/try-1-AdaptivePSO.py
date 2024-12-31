import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particles
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.Inf)
        global_best_position = None
        global_best_value = np.Inf
        evaluations = 0
        
        # PSO parameters
        inertia_weight = 0.9
        cognitive_coeff = 2.0
        social_coeff = 2.0
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
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
            
            # Update inertia weight adaptively
            inertia_weight = 0.4 + (0.9 - 0.4) * (1 - evaluations / self.budget)
            
            # Decay learning parameters
            cognitive_coeff *= 0.99
            social_coeff *= 0.99
            
            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                cognitive_velocity = cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = social_coeff * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt