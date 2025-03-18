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
            exploration_factor = np.random.uniform(0.05, 0.2) * (1 - evaluations / self.budget)  # Changed this line
            inertia_weight = 0.9 - 0.8 * (evaluations / self.budget)
            mutation_scale = 0.5 + 0.4 * np.random.rand()  # New line
            if np.random.rand() < 0.2:  # Changed from 0.15 to 0.2
                trial_solution = np.random.uniform(lb, ub, self.dim)
            else:
                trial_solution = best_solution + inertia_weight * np.random.uniform(-mutation_scale, mutation_scale, self.dim) * (ub - lb) * exploration_factor  # Changed this line
            trial_solution = np.clip(trial_solution, lb, ub)
            trial_value = func(trial_solution)
            evaluations += 1

            # Changed comparison for more robustness
            if trial_value < best_value - 1e-8:  # Changed this line
                best_solution = trial_solution
                best_value = trial_value

        return best_solution

# Usage:
# optimizer = AdaptiveDifferentialRandomWalk(budget, dim)
# best_solution = optimizer(func)