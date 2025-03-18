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

        while evaluations < self.budget:
            exploration_factor = np.random.uniform(0.05, 0.25) * (1 - evaluations / self.budget) ** 2  # Modified line
            inertia_weight = 0.95 - 0.85 * (evaluations / self.budget)  # Modified line
            if np.random.rand() < 0.2:  # Modified from 0.15 to 0.2
                trial_solution = np.random.uniform(lb, ub, self.dim)
            else:
                trial_solution = best_solution + inertia_weight * np.random.uniform(-1, 1, self.dim) * (ub - lb) * exploration_factor
            trial_solution = np.clip(trial_solution, lb, ub)
            trial_value = func(trial_solution)
            evaluations += 1

            if trial_value < best_value:
                best_solution = trial_solution
                best_value = trial_value

        return best_solution