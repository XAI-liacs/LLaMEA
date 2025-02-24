import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Number of initial particles in the swarm
        initial_samples = min(self.budget // 2, 10 * self.dim)
        
        # Initialize particles and their velocities
        particles = np.random.uniform(lb, ub, (initial_samples, self.dim))
        velocities = np.random.uniform(-0.05, 0.05, (initial_samples, self.dim))  # Adjusted velocity scaling
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= initial_samples

        # Update particles towards the best found position
        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]
        
        if self.budget > 0:
            for i in range(initial_samples):
                r1, r2 = np.random.rand(2)
                velocities[i] += r1 * (particles[i] - global_best_position) + r2 * (global_best_position - particles[i])
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)  # Ensure within bounds
                new_eval = func(particles[i])
                if new_eval < sample_evals[i]:
                    sample_evals[i] = new_eval
                    if new_eval < sample_evals[global_best_idx]:
                        global_best_idx = i
                        global_best_position = particles[i]

        def local_optimization(x0):
            nonlocal sample_evals, global_best_idx
            result = minimize(func, x0, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-8, 'fatol': 1e-8})
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val
        
        return global_best_position, sample_evals[global_best_idx]