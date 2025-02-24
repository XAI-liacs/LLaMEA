import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(10, budget // 4)
        self.inertia_weight = 0.5
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.velocity_clamp = 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = self.initialize_particles(bounds)
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        evaluations = self.num_particles
        
        while evaluations < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_constant * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_constant * r2 * (global_best_position - particles[i]))
                
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], bounds[0], bounds[1])
                
                current_value = func(particles[i])
                evaluations += 1
                
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_values[i] = current_value
                    
                    if current_value < global_best_value:
                        global_best_position = particles[i]
                        global_best_value = current_value
            
            # Local search enhancement
            result = minimize(
                func, global_best_position, method='Nelder-Mead',
                options={'maxiter': self.budget - evaluations, 'fatol': 1e-5}
            )
            
            if result.success and result.fun < global_best_value:
                global_best_value = result.fun
                global_best_position = result.x
        
        return global_best_position

    def initialize_particles(self, bounds):
        return np.random.uniform(low=bounds[0], high=bounds[1], size=(self.num_particles, self.dim))