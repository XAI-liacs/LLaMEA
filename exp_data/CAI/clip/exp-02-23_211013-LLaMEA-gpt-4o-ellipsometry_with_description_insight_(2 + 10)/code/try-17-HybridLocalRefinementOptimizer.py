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

            # Adaptive learning rate and stochastic gradient refinement
            learning_rate = 0.1
            for _ in range(5):
                if current_budget >= self.budget:
                    break
                gradient = np.random.randn(self.dim) * learning_rate
                candidate = best_solution + gradient
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                current_budget += 1

                if candidate_score < best_score:
                    best_solution = candidate
                    best_score = candidate_score
                    learning_rate *= 0.9
                else:
                    learning_rate *= 1.1

            # Random walk exploration for additional refinement
            if current_budget < self.budget:
                random_walk_step = np.random.randn(self.dim) * 0.05
                candidate = best_solution + random_walk_step
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                current_budget += 1

                if candidate_score < best_score:
                    best_solution = candidate
                    best_score = candidate_score

        return best_solution