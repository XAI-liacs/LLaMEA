import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        
        # Using PSO for better exploration
        num_particles = min(20, self.budget // 10)
        particles = np.random.uniform(lb, ub, size=(num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, size=(num_particles, self.dim))
        
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best_idx = np.argmin(personal_best_values)
        global_best = particles[global_best_idx]
        
        for _ in range(self.budget // num_particles):
            if self.evaluations >= self.budget:
                break
            
            for i in range(num_particles):
                if self.evaluations >= self.budget:
                    break
                
                # Update velocity and position
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.5 * velocities[i] + 2.0 * r1 * (personal_best[i] - particles[i]) + 2.0 * r2 * (global_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                # Evaluate new positions
                val = func(particles[i])
                self.evaluations += 1
                
                # Update personal and global bests
                if val < personal_best_values[i]:
                    personal_best_values[i] = val
                    personal_best[i] = particles[i]
                    if val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = particles[i]

        # Local optimization using L-BFGS-B starting from global best
        res = minimize(func, global_best, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
        self.evaluations += res.nfev
        
        return res.x