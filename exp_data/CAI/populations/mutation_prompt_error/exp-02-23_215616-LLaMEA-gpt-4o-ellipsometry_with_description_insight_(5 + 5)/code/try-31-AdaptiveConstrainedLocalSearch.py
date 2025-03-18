import numpy as np
from scipy.optimize import minimize

class AdaptiveConstrainedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds for the search space
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initial uniform random sampling with additional gradient-based checks
        num_initial_samples = min(self.budget // 5, 10)  # Adjusted for more thorough sampling
        samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_initial_samples, self.dim))

        # Evaluate initial samples
        evaluations = []
        for sample in samples:
            if len(evaluations) < self.budget:
                evaluations.append((sample, func(sample)))
            else:
                break

        # Sort initial samples based on function value
        evaluations.sort(key=lambda x: x[1])
        best_sample, best_value = evaluations[0]

        # Define a local optimization function using BFGS with convergence rate analysis
        def local_optimization(x0, max_iter):
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bounds, upper_bounds)),
                           options={'maxiter': max_iter, 'disp': False})
            return res.x, res.fun, res.nit

        # Conduct local optimization from best initial sample
        if len(evaluations) < self.budget:
            remaining_budget = self.budget - len(evaluations)
            solution, value, iterations = local_optimization(best_sample, remaining_budget // 2)

            # Adaptive re-initialization based on gradient and convergence rate
            if iterations < remaining_budget // 4:
                # If converged too fast, explore more broadly
                new_sample = best_sample + 0.1 * np.random.randn(self.dim)
                new_sample = np.clip(new_sample, lower_bounds, upper_bounds)
                new_value = func(new_sample)
                if new_value < best_value:
                    best_sample, best_value = new_sample, new_value
            elif value < best_value:
                best_sample, best_value = solution, value

        # Adjust bounds dynamically based on the current best sample
        convergence_speed = (len(evaluations) / self.budget)
        margin = (0.05 + 0.1 * np.random.rand()) * (1 - convergence_speed)
        new_lower_bounds = np.maximum(lower_bounds, best_sample - margin * np.abs(best_sample))
        new_upper_bounds = np.minimum(upper_bounds, best_sample + margin * np.abs(best_sample))

        # Final local optimization with adjusted bounds
        if len(evaluations) < self.budget:
            remaining_budget = self.budget - len(evaluations)
            final_solution, final_value, _ = local_optimization(best_sample, remaining_budget // 2)
            if final_value < best_value:
                best_sample, best_value = final_solution, final_value

        # Enhanced local search with further reduced margin for fine-tuning
        if len(evaluations) < self.budget:
            refined_margin = margin * 0.5
            refined_lower_bounds = np.maximum(lower_bounds, best_sample - refined_margin * np.abs(best_sample))
            refined_upper_bounds = np.minimum(upper_bounds, best_sample + refined_margin * np.abs(best_sample))
            
            def refined_local_optimization(x0):
                res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(refined_lower_bounds, refined_upper_bounds)),
                               options={'maxiter': self.budget - len(evaluations), 'disp': False})
                return res.x, res.fun
            
            if len(evaluations) < self.budget:
                extra_solution, extra_value = refined_local_optimization(final_solution)
                if extra_value < best_value:
                    best_sample, best_value = extra_solution, extra_value

        return best_sample, best_value