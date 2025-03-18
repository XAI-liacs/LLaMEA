import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor

class MultiStartAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(15, self.budget // 4)

        def multi_start_sampling():
            population = np.random.uniform(
                low=func.bounds.lb, 
                high=func.bounds.ub, 
                size=(num_initial_samples, self.dim)
            )
            f_values = np.array([func(ind) for ind in population])
            best_idx = np.argmin(f_values)
            return population[best_idx], f_values[best_idx]
        
        best_initial_sample, best_initial_value = multi_start_sampling()
        remaining_budget = self.budget - num_initial_samples
        
        if remaining_budget > 0:
            def local_objective(x):
                return func(x)
            
            additional_samples = np.random.uniform(
                low=func.bounds.lb, 
                high=func.bounds.ub, 
                size=(6, self.dim)
            )
            additional_f_values = np.array([func(ind) for ind in additional_samples])
            best_additional_idx = np.argmin(additional_f_values)
            best_additional_sample = additional_samples[best_additional_idx]
            
            gbr = GradientBoostingRegressor()
            gbr.fit(additional_samples, additional_f_values)
            predicted_sample = gbr.predict([best_initial_sample])

            starting_points = [best_initial_sample, best_additional_sample] + [
                np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=self.dim) for _ in range(2)
            ] + [np.clip(predicted_sample, func.bounds.lb, func.bounds.ub)]  # Predict within bounds
            
            best_result = {'fun': float('inf')}
            for start in starting_points:
                result = minimize(
                    local_objective, 
                    start, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': remaining_budget // len(starting_points), 'ftol': 1e-9}
                )
                if result.fun < best_result['fun']:
                    best_result = result

            if best_result['fun'] < best_initial_value:
                return best_result.x
        
        return best_initial_sample