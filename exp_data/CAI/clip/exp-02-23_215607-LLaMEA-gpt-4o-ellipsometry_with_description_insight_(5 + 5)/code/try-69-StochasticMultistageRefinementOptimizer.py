import numpy as np
from scipy.optimize import minimize

class StochasticMultistageRefinementOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_initial_samples = min(10, self.budget // 5)  # Initial sampling size
        initial_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_initial_samples, self.dim))
        evals = 0
        
        # Evaluate initial samples and select the top-performing ones
        evaluated_samples = []
        for sample in initial_samples:
            if evals >= self.budget:
                break
            value = func(sample)
            evals += 1
            evaluated_samples.append((sample, value))
        
        # Sort samples by their objective function values
        evaluated_samples.sort(key=lambda x: x[1])
        top_samples = [s for s, v in evaluated_samples[:3]]  # Select the top 3 samples

        # Dynamic subspace refinement and local search
        for sample in top_samples:
            if evals >= self.budget:
                break
            
            def wrapped_func(x):
                nonlocal evals
                if evals >= self.budget:
                    return float('inf')
                value = func(x)
                evals += 1
                return value

            # Local optimization using Nelder-Mead with adjusted bounds
            result = minimize(
                wrapped_func, 
                sample, 
                method='Nelder-Mead', 
                options={'maxfev': max(1, (self.budget - evals) // len(top_samples))}
            )
            
            if result.success and result.fun < evaluated_samples[0][1]:
                evaluated_samples[0] = (result.x, result.fun)

        return evaluated_samples[0][0]