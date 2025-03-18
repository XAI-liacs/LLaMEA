import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundsShrinkingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        bounds = np.array(list(zip(lb, ub)))

        best_solution = None
        best_value = float('inf')
        
        initial_sample_budget = max(10, int(0.15 * self.budget))
        local_search_budget = int(0.1 * self.budget)
        remaining_budget = self.budget - initial_sample_budget - local_search_budget

        initial_guesses = np.random.uniform(lb, ub, (initial_sample_budget, self.dim))
        initial_guesses = np.append(initial_guesses, [np.linspace(lb, ub, self.dim)], axis=0)

        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        refinement_factor = 0.15
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        def local_search_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': local_search_budget})
            return res.x, res.fun
        
        refined_solution, refined_value = local_search_optimization(best_solution)
        refined_solution, refined_value = local_search_optimization(refined_solution)

        further_refinement_factor = 0.05
        further_refined_bounds = np.array([
            [
                max(l, refined_solution[i] - further_refinement_factor * (u - l)),
                min(u, refined_solution[i] + further_refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(refined_bounds)
        ])

        stochastic_search_budget = remaining_budget
        stochastic_guesses = refined_solution + np.random.uniform(-0.05, 0.05, (stochastic_search_budget, self.dim))
        stochastic_guesses = np.clip(stochastic_guesses, lb, ub)

        for guess in stochastic_guesses:
            value = func(guess)
            if value < refined_value:
                refined_value = value
                refined_solution = guess

        return refined_solution if refined_value < best_value else best_solution