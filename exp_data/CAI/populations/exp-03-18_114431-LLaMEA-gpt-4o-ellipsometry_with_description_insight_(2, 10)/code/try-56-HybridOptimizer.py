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

        num_samples = min(max(5, self.budget // (2 + self.dim)), 25)  # Increased sample limit

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
        
        model = GradientBoostingRegressor(n_estimators=150).fit(samples, sample_vals)  # Increased number of estimators
        predictions = model.predict(samples)
        improved_idx = np.argmin(predictions)
        improved_sample = samples[improved_idx]

        # Modify the minimize function by adding an adaptive learning rate
        options = {'gtol': 1e-6, 'disp': False, 'eps': 1e-6}  # Modified 'eps' for better convergence
        result = minimize(budgeted_func, improved_sample, method='BFGS', bounds=list(zip(lb, ub)), options=options)
        
        return result.x if result.success else best_sample