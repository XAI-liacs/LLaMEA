import numpy as np
from scipy.optimize import minimize

class ImprovedPhotonicsOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Retrieve bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Number of initial samples
        num_samples = max(5, self.budget // 10)
        
        # Generate initial adaptive random samples
        initial_samples = lb + (ub - lb) * np.random.rand(num_samples, self.dim)
        
        best_solution = None
        best_score = float('inf')
        
        # Evaluation counter
        evaluations = 0

        for sample in initial_samples:
            # Convert bounds to a format compatible with minimize
            bounds = [(lb[i], ub[i]) for i in range(self.dim)]
            
            # Local optimization using a dynamically adjusted Nelder-Mead
            result = minimize(func, sample, method='Nelder-Mead', bounds=bounds, options={'maxfev': min(self.budget - evaluations, 100)})
            
            evaluations += result.nfev
            
            # Update best solution if improved
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x

            # Use quadratic interpolation to refine the search around the best solution
            if best_solution is not None:
                new_samples = self.quadratic_interpolation(best_solution, lb, ub, 3)
                for new_sample in new_samples:
                    if evaluations >= self.budget:
                        break
                    result = minimize(func, new_sample, method='Nelder-Mead', bounds=bounds, options={'maxfev': min(self.budget - evaluations, 100)})
                    evaluations += result.nfev
                    if result.fun < best_score:
                        best_score = result.fun
                        best_solution = result.x
                
                # Dynamically adjust bounds based on current best solution
                lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
                ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))
            
            # Check if budget is exceeded
            if evaluations >= self.budget:
                break
        
        return best_solution

    def quadratic_interpolation(self, best_solution, lb, ub, num_points):
        # Generate points around the best solution using quadratic interpolation
        perturbation = np.linspace(-0.05, 0.05, num_points) * (ub - lb)
        new_samples = [np.clip(best_solution + delta, lb, ub) for delta in perturbation]
        return new_samples