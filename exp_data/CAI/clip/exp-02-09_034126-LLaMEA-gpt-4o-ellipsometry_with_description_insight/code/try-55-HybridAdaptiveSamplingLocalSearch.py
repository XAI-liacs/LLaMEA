import numpy as np
from scipy.optimize import minimize

class HybridAdaptiveSamplingLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        initial_samples = self.adaptive_sampling(bounds)
        remaining_budget = self.budget - len(initial_samples)
        
        best_solution = None
        best_cost = float('inf')

        for sample in initial_samples:
            local_result, used_budget = self.hybrid_local_optimization(func, sample, bounds, remaining_budget)
            remaining_budget -= used_budget
            if local_result.fun < best_cost:
                best_solution = local_result.x
                best_cost = local_result.fun

            if remaining_budget <= 0:
                break

        return best_solution

    def adaptive_sampling(self, bounds):
        num_samples = min(self.budget // 10, 12)  # 10% of budget for sampling
        samples = []
        for _ in range(num_samples):
            sample = [np.random.uniform(low, high) for low, high in bounds]
            samples.append(sample)
        return samples

    def hybrid_local_optimization(self, func, initial_guess, bounds, remaining_budget):
        exploration_budget = max(remaining_budget // 2, 1)
        exploitation_budget = remaining_budget - exploration_budget

        # Exploration phase
        explore_result = minimize(func, initial_guess, method='Nelder-Mead', options={'maxiter': exploration_budget, 'disp': False})
        
        # Exploitation phase
        exploit_result = minimize(func, explore_result.x, method='L-BFGS-B', bounds=bounds, options={'maxiter': exploitation_budget, 'disp': False})
        
        total_used_budget = explore_result.nfev + exploit_result.nfev
        return exploit_result, total_used_budget