import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class GaussianProcessHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        bounds = np.array(list(zip(lb, ub)))

        # Initialize variables
        best_solution = None
        best_value = float('inf')
        initial_sample_budget = max(10, int(0.2 * self.budget))
        gp_budget = self.budget - initial_sample_budget

        # Initial uniform sampling for exploration
        initial_guesses = np.random.uniform(lb, ub, (initial_sample_budget, self.dim))
        initial_values = np.array([func(guess) for guess in initial_guesses])

        # Determine best initial solution
        for i, value in enumerate(initial_values):
            if value < best_value:
                best_value = value
                best_solution = initial_guesses[i]

        # Gaussian Process Regression on initial samples
        kernel = RBF(length_scale=np.std(initial_guesses, axis=0))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gp.fit(initial_guesses, initial_values)

        # Adaptive sampling with GP predictions
        for _ in range(gp_budget):
            # Predict uncertainty and mean of the unexplored space
            sample_points = np.random.uniform(lb, ub, (100, self.dim))
            mean, std = gp.predict(sample_points, return_std=True)
            acquisition = mean - std  # Exploitation guided by lower confidence bound

            # Select point with the best acquisition score
            next_guess = sample_points[np.argmin(acquisition)]
            next_value = func(next_guess)

            # Update GP model and best solution
            gp.fit(np.vstack((initial_guesses, next_guess)), np.append(initial_values, next_value))
            if next_value < best_value:
                best_value = next_value
                best_solution = next_guess

        # Final refinement with BFGS in local area
        refinement_factor = 0.1
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': gp_budget})
            return res.x, res.fun

        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution