import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        num_samples = min(max(5, self.budget // 4), 15)

        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_vals = [func(sample) for sample in samples]

        self.budget -= num_samples

        best_idx = np.argmin(sample_vals)
        best_sample = samples[best_idx]
        
        def budgeted_func(x):
            if self.budget <= 0:
                raise RuntimeError("Budget exceeded")
            self.budget -= 1
            return func(x)
        
        # Predictive model for enhancing initial guesses
        model = Ridge().fit(samples, sample_vals)  # Changed line from GradientBoostingRegressor to Ridge
        predictions = model.predict(samples)
        improved_idx = np.argmin(predictions)
        improved_sample = samples[improved_idx]

        result = minimize(budgeted_func, improved_sample, method='BFGS', bounds=list(zip(lb, ub)))
        
        return result.x if result.success else best_sample