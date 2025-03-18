import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

class GradientInformedAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def uniform_sampling(self, bounds, num_samples):
        samples = []
        for lb, ub in zip(bounds.lb, bounds.ub):
            samples.append(np.random.uniform(lb, ub, num_samples))
        return np.array(samples).T

    def gradient_guided_sampling(self, func, x, bounds, epsilon=1e-8):
        grad = approx_fprime(x, func, epsilon)
        norm_grad = grad / (np.linalg.norm(grad) + 1e-8)
        sampled_points = []
        for lb, ub in zip(bounds.lb, bounds.ub):
            delta = (ub - lb) * 0.1  # 10% of the range
            new_point = np.clip(x - delta * norm_grad, lb, ub)
            sampled_points.append(new_point)
        return np.array(sampled_points).T

    def local_optimization(self, func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun

    def __call__(self, func):
        num_initial_samples = min(10, self.budget // 2)
        initial_points = self.uniform_sampling(func.bounds, num_initial_samples)
        
        best_solution = None
        best_value = float('inf')

        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        for x0 in initial_points:
            if self.budget <= 0:
                break

            # Use local optimization from the initial point
            x, value = self.local_optimization(func, x0, bounds)
            self.budget -= 1  # Counting the local optimization as a single budget usage
            
            # If budget allows, refine search using gradient information
            if self.budget > 0:
                guided_points = self.gradient_guided_sampling(func, x, bounds)
                for guided_x0 in guided_points:
                    if self.budget <= 0:
                        break
                    guided_x, guided_value = self.local_optimization(func, guided_x0, bounds)
                    self.budget -= 1  # Counting each trial as a budget usage
                    if guided_value < best_value:
                        best_value = guided_value
                        best_solution = guided_x

            if value < best_value:
                best_value = value
                best_solution = x

        return best_solution