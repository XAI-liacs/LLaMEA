import numpy as np
from scipy.optimize import minimize, differential_evolution

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Decrease initial samples for more budget on DE
        initial_samples = min(self.budget // 3, 8 * self.dim)

        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        self.budget -= initial_samples

        best_idx = np.argmin(sample_evals)
        best_sample = samples[best_idx]
        best_val = sample_evals[best_idx]

        # Use Differential Evolution for better exploration
        if self.budget > 0:
            result = differential_evolution(func, bounds=list(zip(lb, ub)), maxiter=self.budget // 2,
                                            popsize=self.dim, disp=False)
            if result.fun < best_val:
                best_sample, best_val = result.x, result.fun

        def local_optimization(x0):
            nonlocal best_val, best_sample
            result = minimize(func, x0, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-8, 'fatol': 1e-8})
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = local_optimization(best_sample)
            if final_val < best_val:
                best_sample, best_val = final_sample, final_val
        
        return best_sample, best_val