import numpy as np
from scipy.optimize import minimize

class GradientInformedSamplingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Retrieve bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Step 1: Initial Uniform Sampling and Gradient Approximation
        initial_samples = min(self.budget // 4, 40 * self.dim)
        samples = np.random.uniform(low=lb, high=ub, size=(initial_samples, self.dim))
        evaluations = []
        
        for s in samples:
            if self.evaluations >= self.budget:
                break
            eval_result = func(s)
            evaluations.append((eval_result, s))
            self.evaluations += 1
        
        # Sort and select top performers
        evaluations.sort(key=lambda x: x[0])
        top_samples = [e[1] for e in evaluations[:5]]
        
        # Step 2: Gradient-Informed Sampling and BFGS Optimization
        best_sample = None
        best_value = float('inf')
        for sample in top_samples:
            if self.evaluations < self.budget:
                # Compute numerical gradient
                epsilon = 1e-5
                grad = np.zeros(self.dim)
                for i in range(self.dim):
                    s1 = np.copy(sample)
                    s1[i] += epsilon
                    f1 = func(s1)
                    grad[i] = (f1 - eval_result) / epsilon
                
                # Generate gradient-informed sample points
                grad_samples = [sample + 0.1 * grad, sample - 0.1 * grad]
                for g_sample in grad_samples:
                    g_sample = np.clip(g_sample, lb, ub)
                    if self.evaluations < self.budget:
                        eval_g_result = func(g_sample)
                        self.evaluations += 1
                        if eval_g_result < best_value:
                            best_sample = g_sample
                            best_value = eval_g_result
                
                # Local BFGS optimization
                local_budget = self.budget - self.evaluations
                options = {'maxiter': local_budget, 'gtol': 1e-9}
                result = minimize(func, best_sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
                if result.success and result.fun < best_value:
                    best_sample = result.x
                    best_value = result.fun

        return best_sample if best_sample is not None else top_samples[0]