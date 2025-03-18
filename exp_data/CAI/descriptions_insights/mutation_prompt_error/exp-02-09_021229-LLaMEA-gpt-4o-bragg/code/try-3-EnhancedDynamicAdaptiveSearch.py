import numpy as np

class EnhancedDynamicAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        step_size = (ub - lb) / 10
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_solution)
        inertia_weight = 0.9

        for _ in range(self.budget - 1):
            inertia_weight = max(0.4, inertia_weight * 0.99)  # Dynamic inertia weight decay
            candidate_solution = (current_solution +
                                  inertia_weight * np.random.uniform(-step_size, step_size, self.dim))
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
                step_size *= 1.3  # Slightly larger increase multiplier
            else:
                step_size *= 0.8  # Slightly larger decrease multiplier

            # Local exploitation
            if np.random.rand() < 0.2:  # 20% chance for local exploitation
                perturbation = (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
                candidate_solution = np.clip(current_solution + perturbation, lb, ub)
                candidate_value = func(candidate_solution)
                if candidate_value < current_value:
                    current_solution = candidate_solution
                    current_value = candidate_value

            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

        return self.best_solution, self.best_value