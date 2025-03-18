import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        num_samples = min(max(5, self.budget // (2 + self.dim)), 20)

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
        
        model = GradientBoostingRegressor().fit(samples, sample_vals)
        predictions = model.predict(samples)
        improved_idx = np.argmin(predictions)
        improved_sample = samples[improved_idx]

        # Modify options to include adaptive momentum
        options = {'gtol': 1e-6, 'disp': False, 'eps': 1e-8, 'momentum': 0.9}  # Added 'momentum' for adaptive momentum
        result = minimize(budgeted_func, improved_sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
        
        return result.x if result.success else best_sample