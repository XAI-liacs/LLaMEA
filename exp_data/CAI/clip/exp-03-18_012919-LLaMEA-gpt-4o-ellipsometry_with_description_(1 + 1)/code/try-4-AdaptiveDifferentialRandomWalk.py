import numpy as np

class AdaptiveDifferentialRandomWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, self.dim)
        best_value = func(best_solution)
        evaluations = 1

        def levy_flight(Lambda):
            sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
                     (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.abs(v) ** (1 / Lambda)
            return 0.01 * step  # Small scaling factor

        while evaluations < self.budget:
            exploration_factor = np.random.uniform(0.05, 0.3) * (1 - evaluations / self.budget)  # Changed exploration range
            inertia_weight = 0.9 - 0.8 * (evaluations / self.budget)
            if np.random.rand() < 0.15:
                trial_solution = np.random.uniform(lb, ub, self.dim) + levy_flight(1.5)  # Levy flight step
            else:
                trial_solution = best_solution + inertia_weight * np.random.uniform(-1, 1, self.dim) * (ub - lb) * exploration_factor
            trial_solution = np.clip(trial_solution, lb, ub)
            trial_value = func(trial_solution)
            evaluations += 1

            if trial_value < best_value:
                best_solution = trial_solution
                best_value = trial_value

        return best_solution

# Usage:
# optimizer = AdaptiveDifferentialRandomWalk(budget, dim)
# best_solution = optimizer(func)