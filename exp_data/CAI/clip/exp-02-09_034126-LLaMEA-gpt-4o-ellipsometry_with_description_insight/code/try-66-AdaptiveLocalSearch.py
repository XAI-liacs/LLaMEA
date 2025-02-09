import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Halton

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        initial_guess = self.uniform_sampling(bounds)
        remaining_budget = self.budget - len(initial_guess)
        
        best_solution = None
        best_cost = float('inf')

        for guess in initial_guess:
            result = self.local_optimization(func, guess, bounds, remaining_budget)
            if result.fun < best_cost:
                best_solution = result.x
                best_cost = result.fun
                remaining_budget -= result.nfev

            if remaining_budget <= 0:
                break

        return best_solution

    def uniform_sampling(self, bounds):
        num_samples = min(self.budget // 8, 10)
        halton_sampler = Halton(d=self.dim)  # Use Halton sequence for initial sampling
        samples = halton_sampler.random(n=num_samples)
        scaled_samples = [bounds[i][0] + s * (bounds[i][1] - bounds[i][0]) for i, s in enumerate(samples.T)]
        return np.array(scaled_samples).T.tolist()

    def local_optimization(self, func, initial_guess, bounds, remaining_budget):
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options=options)
        return result