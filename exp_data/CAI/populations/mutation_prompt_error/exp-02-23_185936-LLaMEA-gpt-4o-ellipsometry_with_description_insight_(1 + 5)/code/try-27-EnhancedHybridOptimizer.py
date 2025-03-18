import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_particles = min(self.budget // 4, 20)
        inertia_weight = 0.5
        cognitive_weight = 1.5
        social_weight = 1.5
        
        # Initialize particles
        particles = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_particles])
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_position = particles[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = num_particles

        while evaluations < self.budget:
            if evaluations + num_particles > self.budget:
                num_particles = self.budget - evaluations

            # Update particle velocities and positions
            for i in range(num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    inertia_weight * velocities[i] +
                    cognitive_weight * r1 * (personal_best_positions[i] - particles[i]) +
                    social_weight * r2 * (global_best_position - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], [b[0] for b in bounds], [b[1] for b in bounds])
                
                # Evaluate new position
                current_value = func(particles[i])
                evaluations += 1

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_values[i] = current_value

                # Update global best
                if current_value < global_best_value:
                    global_best_position = particles[i]
                    global_best_value = current_value

            # Use a local optimizer for the best particle found
            result = minimize(func, global_best_position, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations})
            evaluations += result.nfev

            # Update global best if necessary
            if result.fun < global_best_value:
                global_best_value = result.fun
                global_best_position = result.x

        return global_best_position