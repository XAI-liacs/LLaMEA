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
        no_improvement_counter = 0

        while evaluations < self.budget:
            exploration_factor = (0.1 + 0.1 * np.sin(5 * np.pi * evaluations / self.budget)) * np.exp(-evaluations / (0.1 * self.budget))
            inertia_weight = (0.8 + 0.1 * np.cos(4 * np.pi * evaluations / self.budget)) * np.exp(-evaluations / (0.05 * self.budget))
            step_size = 0.5 * np.random.rand() + 0.5  # Changed line
            if np.random.rand() < 0.10:
                trial_solution = np.random.uniform(lb, ub, self.dim)
            else:
                improvement_scale = 1 - (no_improvement_counter / (self.budget * 0.05))
                trial_solution = best_solution + inertia_weight * improvement_scale * np.random.uniform(-1, 1, self.dim) * (ub - lb) * exploration_factor * step_size  # Changed line
            trial_solution = np.clip(trial_solution, lb, ub)
            trial_value = func(trial_solution)
            evaluations += 1

            if trial_value < best_value:
                best_solution = trial_solution
                best_value = trial_value
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                if no_improvement_counter > self.budget * 0.06:
                    trial_solution = np.random.uniform(lb, ub, self.dim)
                    if np.random.rand() < 0.2:  # Changed line
                        best_solution = trial_solution  # Changed line
                        best_value = func(best_solution)  # Changed line
                        no_improvement_counter = 0  # Changed line

        return best_solution