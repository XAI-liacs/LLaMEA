import numpy as np
from scipy.optimize import minimize

class HybridGradientPerturbationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initial uniform sampling
        initial_samples = min(self.budget // 3, 10 * self.dim)
        particles = lb + (ub - lb) * np.random.rand(initial_samples, self.dim)
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= initial_samples

        # Identify the best initial sample
        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]

        if self.budget > 0:
            # Stochastic gradient estimation with perturbation
            for _ in range(min(self.budget // 2, initial_samples)):
                perturbation = np.random.uniform(-0.05, 0.05, self.dim)
                candidate = global_best_position + perturbation
                candidate = np.clip(candidate, lb, ub)

                candidate_eval = func(candidate)
                self.budget -= 1
                if candidate_eval < sample_evals[global_best_idx]:
                    global_best_position, sample_evals[global_best_idx] = candidate, candidate_eval

        def local_optimization(x0):
            nonlocal sample_evals, global_best_idx
            # Gradient-based local refinement
            result = minimize(func, x0, method='BFGS',
                              options={'maxiter': self.budget, 'gtol': 1e-6})
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val
        
        return global_best_position, sample_evals[global_best_idx]