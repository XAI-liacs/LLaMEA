import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from concurrent.futures import ThreadPoolExecutor

class ParallelSobolOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_optimize(self, func, point, lb, ub):
        res = minimize(func, point, method='L-BFGS-B', bounds=zip(lb, ub))
        return res.fun, res.x, res.nfev

    def __call__(self, func):
        # Define the search space
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialize variables
        current_budget = 0
        best_solution = None
        best_score = float('inf')
        
        # Use Sobol sequence for initial points to improve coverage
        sampler = Sobol(d=self.dim, scramble=True)
        num_initial_points = min(self.budget, 16)  # Adjusted based on budget
        initial_points = lb + (ub - lb) * sampler.random(num_initial_points)
        
        # Prepare for parallel execution
        with ThreadPoolExecutor(max_workers=min(8, len(initial_points))) as executor:
            futures = [executor.submit(self._local_optimize, func, point, lb, ub) for point in initial_points]

            for future in futures:
                if current_budget >= self.budget:
                    break

                score, solution, nfev = future.result()
                current_budget += nfev
                
                # Update the best solution found
                if score < best_score:
                    best_solution = solution
                    best_score = score

        return best_solution