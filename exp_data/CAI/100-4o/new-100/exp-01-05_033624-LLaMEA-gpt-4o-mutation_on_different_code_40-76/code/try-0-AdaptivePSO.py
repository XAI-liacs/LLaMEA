import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particle positions and velocities
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.Inf)
        
        global_best_value = np.Inf
        global_best_position = None
        
        # PSO parameters
        inertia_weight = 0.9
        cognitive_component = 2.0
        social_component = 2.0
        
        for _ in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                # Evaluate function at each particle's position
                f_value = func(positions[i])
                
                # Update personal best
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i]
                
                # Update global best
                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = positions[i]
                    
            # Update velocities and positions
            for i in range(self.num_particles):
                inertia_weight = 0.4 + 0.5 * (global_best_value - personal_best_values[i]) / (global_best_value + 1e-10)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i] +
                                cognitive_component * r1 * (personal_best_positions[i] - positions[i]) +
                                social_component * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt