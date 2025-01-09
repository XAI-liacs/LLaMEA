import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.Inf)
        
        global_best_value = np.Inf
        global_best_position = None
        elite_particle_index = 0
        
        # Adaptive PSO parameters
        inertia_weight = 0.9
        cognitive_component = 1.5
        social_component = 1.5
        velocity_clamp = 0.5
        
        for _ in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                f_value = func(positions[i])
                
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i]
                
                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = positions[i]
                    elite_particle_index = i
            
            # Dynamic adaptation of parameters
            if _ > 0 and _ % 10 == 0:
                inertia_weight *= 0.99
                if global_best_value < np.median(personal_best_values):
                    cognitive_component *= 1.05
                    social_component *= 0.95
                else:
                    cognitive_component *= 0.95
                    social_component *= 1.05
            
            # Update velocities and positions
            for i in range(self.num_particles):
                inertia_weight = max(0.4, inertia_weight)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i] +
                                cognitive_component * r1 * (personal_best_positions[i] - positions[i]) +
                                social_component * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt