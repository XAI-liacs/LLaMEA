import numpy as np
from scipy.optimize import minimize

class MultiStageProgressiveRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        global_samples = self.global_exploration(bounds)
        remaining_budget = self.budget - len(global_samples)

        best_solution = None
        best_cost = float('inf')

        for guess in global_samples:
            result = self.local_refinement(func, guess, bounds, remaining_budget // len(global_samples))
            if result.fun < best_cost:
                best_solution = result.x
                best_cost = result.fun

            if remaining_budget <= 0:
                break

        return best_solution

    def global_exploration(self, bounds):
        num_samples = max(5, self.budget // 20)  # Use a fraction of budget for initial global exploration
        samples = []
        for _ in range(num_samples):
            sample = [np.random.uniform(low, high) for low, high in bounds]
            samples.append(sample)
        return samples

    def local_refinement(self, func, initial_guess, bounds, local_budget):
        resolution_levels = [0.1, 0.01, 0.001]  # Progressive refinement
        current_guess = initial_guess
        for resolution in resolution_levels:
            local_bounds = self.narrow_down_bounds(current_guess, bounds, resolution)
            options = {'maxiter': local_budget // len(resolution_levels), 'disp': False}
            result = minimize(func, current_guess, method='L-BFGS-B', bounds=local_bounds, options=options)
            current_guess = result.x
            if result.nfev >= local_budget:
                break
        return result

    def narrow_down_bounds(self, guess, bounds, resolution):
        new_bounds = []
        for g, (low, high) in zip(guess, bounds):
            span = (high - low) * resolution
            new_low = max(low, g - span / 2)
            new_high = min(high, g + span / 2)
            new_bounds.append((new_low, new_high))
        return new_bounds