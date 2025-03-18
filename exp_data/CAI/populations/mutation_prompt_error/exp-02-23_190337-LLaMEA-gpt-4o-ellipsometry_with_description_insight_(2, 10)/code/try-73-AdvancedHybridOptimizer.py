import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        bounds = np.array(list(zip(lb, ub)))
        best_solution = None
        best_value = float('inf')
        
        initial_sample_budget = max(10, int(0.15 * self.budget))
        exploration_budget = int(0.1 * self.budget)
        optimizer_budget = self.budget - initial_sample_budget - exploration_budget

        initial_guesses = np.random.uniform(lb, ub, (initial_sample_budget, self.dim))  # Standard sampling range

        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess

        exploration_variance = np.var(initial_guesses, axis=0)
        kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(initial_guesses, [func(x) for x in initial_guesses])
        
        exploration_guesses = gp.sample_y(best_solution.reshape(1, -1), n_samples=exploration_budget)
        exploration_guesses = np.clip(exploration_guesses, lb, ub)

        for guess in exploration_guesses:
            value = func(guess.flatten())
            if value < best_value:
                best_value = value
                best_solution = guess.flatten()

        refinement_factor_max = 0.15
        refinement_factor = min(refinement_factor_max, 0.2 * np.linalg.norm(exploration_variance))
        refined_bounds = np.array([
            [
                max(l, best_solution[i] - refinement_factor * (u - l)),
                min(u, best_solution[i] + refinement_factor * (u - l))
            ] for i, (l, u) in enumerate(bounds)
        ])

        def bfgs_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=refined_bounds, options={'maxfun': optimizer_budget})
            return res.x, res.fun

        final_solution, final_value = bfgs_optimization(best_solution)

        return final_solution if final_value < best_value else best_solution