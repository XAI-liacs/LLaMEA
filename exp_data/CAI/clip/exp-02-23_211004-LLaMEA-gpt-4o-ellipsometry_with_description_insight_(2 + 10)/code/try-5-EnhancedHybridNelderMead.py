import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedHybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        
        # Use Latin Hypercube Sampling for diverse initial sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        num_initial_samples = min(10, self.budget // 2)
        sample = sampler.random(num_initial_samples)
        initial_points = qmc.scale(sample, bounds[0], bounds[1])
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for point in initial_points:
            if evaluations >= self.budget:
                break
            # Perform local optimization with dynamic Nelder-Mead
            result = minimize(func, point, method='Nelder-Mead', options={'maxfev': self.budget - evaluations})
            evaluations += result.nfev

            # Update the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Adaptive restart if stuck in a local minimum
            if result.success and evaluations < self.budget:
                new_start_point = np.random.uniform(bounds[0], bounds[1])
                result = minimize(func, new_start_point, method='Nelder-Mead', options={'maxfev': self.budget - evaluations})
                evaluations += result.nfev

                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
        
        return best_solution