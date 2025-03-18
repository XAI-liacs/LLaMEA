import numpy as np
from scipy.optimize import minimize

class AdaptiveConstrainedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        num_initial_samples = min(self.budget // 5, 10)
        samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_initial_samples, self.dim))

        evaluations = []
        for sample in samples:
            if len(evaluations) < self.budget:
                evaluations.append((sample, func(sample)))
            else:
                break

        evaluations.sort(key=lambda x: x[1])
        best_sample, best_value = evaluations[0]

        def local_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bounds, upper_bounds)),
                           options={'maxiter': self.budget - len(evaluations)})
            return res.x, res.fun

        # Changed: Hybrid BFGS and Nelder-Mead for local optimization
        def hybrid_optimization(x0):
            if len(evaluations) < self.budget:
                res_nm = minimize(func, x0, method='Nelder-Mead', options={'maxiter': self.budget // 4})
                if res_nm.fun < best_value:
                    return res_nm.x, res_nm.fun
            return local_optimization(x0)

        if len(evaluations) < self.budget:
            solution, value = hybrid_optimization(best_sample)
            if value < best_value:
                best_sample, best_value = solution, value

        # Changed: Adaptive margin based on the gradient of a solution
        gradient = np.abs(best_sample - samples.mean(axis=0))
        margin = 0.05 + 0.05 * gradient.mean()
        
        new_lower_bounds = np.maximum(lower_bounds, best_sample - margin * best_sample)
        new_upper_bounds = np.minimum(upper_bounds, best_sample + margin * best_sample)

        if len(evaluations) < self.budget:
            final_solution, final_value = hybrid_optimization(best_sample)
            if final_value < best_value:
                best_sample, best_value = final_solution, final_value

        if len(evaluations) < self.budget:
            extra_solution, extra_value = hybrid_optimization(final_solution)
            if extra_value < best_value:
                best_sample, best_value = extra_solution, extra_value

        return best_sample, best_value