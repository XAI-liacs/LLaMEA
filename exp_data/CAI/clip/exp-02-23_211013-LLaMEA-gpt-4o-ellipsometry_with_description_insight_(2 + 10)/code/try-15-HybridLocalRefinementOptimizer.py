import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

class HybridLocalRefinementOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the search space
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialize variables
        current_budget = 0
        best_solution = None
        best_score = float('inf')
        stagnation_threshold = 5  # New line: Establish a stagnation threshold
        stagnation_counter = 0

        # Use Latin Hypercube Sampling to initialize points
        lhs_sampler = LatinHypercube(d=self.dim)
        initial_points = lhs_sampler.random(n=10) * (ub - lb) + lb

        for point in initial_points:
            if current_budget >= self.budget:
                break

            # Perform local optimization using BFGS with bounds
            res = minimize(func, point, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            current_budget += res.nfev

            # Update the best solution found
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1

            # Dynamic region shrinkage to refine search around best solution
            if current_budget < self.budget and best_solution is not None:
                shrink_factor = 0.5
                new_lb = np.maximum(lb, best_solution - shrink_factor * (ub - lb) / 2)
                new_ub = np.minimum(ub, best_solution + shrink_factor * (ub - lb) / 2)
                if np.any(new_lb >= new_ub):
                    continue
                res = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(new_lb, new_ub)))
                current_budget += res.nfev

                # Update the best solution found
                if res.fun < best_score:
                    best_solution = res.x
                    best_score = res.fun
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1

            # Restart if stagnation occurs
            if stagnation_counter >= stagnation_threshold:
                initial_points = lhs_sampler.random(n=5) * (ub - lb) + lb  # Restart with new initial points
                stagnation_counter = 0

        return best_solution