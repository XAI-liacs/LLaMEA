import numpy as np

class DynamicAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        step_size = (ub - lb) / 10  # Initial step size as 1/10 of the search space range
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_solution)

        for _ in range(self.budget - 1):  
            candidate_solution = current_solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                current_solution = 0.7 * candidate_solution + 0.3 * current_solution  # Line changed
                current_value = candidate_value
                step_size *= 1.2  # Increase step size if improvement
            else:
                step_size *= 0.9  # Decrease step size if no improvement

            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

        return self.best_solution, self.best_value

# Example usage:
# optimizer = DynamicAdaptiveSearch(budget=100, dim=5)
# best_solution, best_value = optimizer(your_func)