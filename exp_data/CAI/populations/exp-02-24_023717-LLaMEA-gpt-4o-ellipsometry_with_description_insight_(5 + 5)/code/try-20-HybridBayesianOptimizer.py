import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize

class HybridBayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Step 1: Initial exploration via Bayesian Optimization
        def skopt_func(x):
            nonlocal evaluations
            evaluations += 1
            if evaluations > self.budget:
                return float('inf')
            return func(np.array(x))

        space = [(low, high) for low, high in zip(lb, ub)]
        initial_points = min(5, self.budget // 4)
        res = gp_minimize(
            skopt_func,
            space,
            n_calls=initial_points,
            n_initial_points=initial_points,
            random_state=42
        )
        
        best_solution, best_value = res.x, res.fun

        # Step 2: Hybrid local search with adaptive bounds
        while evaluations < self.budget:
            # Random perturbation to escape local minima
            perturbed_solution = best_solution + np.random.uniform(-0.05, 0.05, self.dim) * (ub - lb)
            perturbed_solution = np.clip(perturbed_solution, lb, ub)

            result = self._local_optimize(func, perturbed_solution, lb, ub)
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Adaptively shrink bounds around the new best solution
                lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
                ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))
            else:
                if evaluations >= self.budget * 0.75:
                    break

        return best_solution

    def _local_optimize(self, func, start_point, lb, ub):
        return minimize(
            func,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxfun': self.budget}
        )