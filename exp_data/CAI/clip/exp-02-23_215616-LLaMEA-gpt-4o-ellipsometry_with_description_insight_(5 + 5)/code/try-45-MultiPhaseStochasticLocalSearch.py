import numpy as np
from scipy.optimize import minimize

class MultiPhaseStochasticLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Phase 1: Stochastic Sampling with Adaptive Sample Size
        phase1_samples = min(self.budget // 3, 30)
        samples = np.random.uniform(lower_bounds, upper_bounds, size=(phase1_samples, self.dim))

        evaluations = []
        for sample in samples:
            if len(evaluations) < self.budget:
                evaluations.append((sample, func(sample)))
            else:
                break

        evaluations.sort(key=lambda x: x[1])
        best_sample, best_value = evaluations[0]

        # Phase 2: Gradient-Based Local Search
        def gradient_local_optimization(x0, max_iter):
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bounds, upper_bounds)),
                           options={'maxiter': max_iter})
            return res.x, res.fun

        if len(evaluations) < self.budget:
            remaining_budget = int(self.budget - len(evaluations))
            solution, value = gradient_local_optimization(best_sample, remaining_budget // 2)
            if value < best_value:
                best_sample, best_value = solution, value

        # Phase 3: Adaptive Margin Reduction and Fine-Tuning
        adaptive_margin = (0.02 + 0.08 * np.random.rand())
        new_lower_bounds = np.maximum(lower_bounds, best_sample - adaptive_margin * np.abs(best_sample))
        new_upper_bounds = np.minimum(upper_bounds, best_sample + adaptive_margin * np.abs(best_sample))

        # Refined Local Optimization
        def refined_gradient_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(new_lower_bounds, new_upper_bounds)),
                           options={'maxiter': self.budget - len(evaluations)})
            return res.x, res.fun

        if len(evaluations) < self.budget:
            final_solution, final_value = refined_gradient_optimization(best_sample)
            if final_value < best_value:
                best_sample, best_value = final_solution, final_value

        return best_sample, best_value