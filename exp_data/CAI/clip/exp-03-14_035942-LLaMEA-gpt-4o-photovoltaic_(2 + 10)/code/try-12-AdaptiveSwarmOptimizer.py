import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        num_particles = 50
        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.zeros((num_particles, self.dim))
        personal_best = positions.copy()
        personal_best_values = np.array([func(pos) for pos in personal_best])
        global_best_index = np.argmin(personal_best_values)
        global_best = personal_best[global_best_index].copy()
        global_best_value = personal_best_values[global_best_index]
        
        w = 0.5 + np.random.rand() / 2
        c1, c2 = 1.5, 1.5
        eval_count = num_particles

        while eval_count < self.budget:
            for i in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - positions[i]) +
                                 c2 * r2 * (global_best - positions[i]))
                
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
                
                value = func(positions[i])
                eval_count += 1
                
                if value < personal_best_values[i]:
                    personal_best[i] = positions[i].copy()
                    personal_best_values[i] = value
                    
                    if value < global_best_value:
                        global_best = positions[i].copy()
                        global_best_value = value

                if eval_count >= self.budget:
                    break
            
            w = 0.5 + np.random.rand() / 2  # Adaptive inertia weight
            c1, c2 = np.random.uniform(1.3, 2.0, 2)  # Dynamic learning factors
            num_particles = min(100, num_particles + 1)  # Increment particle count

        return global_best