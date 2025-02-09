import numpy as np

class AdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.success_rate = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_solution)
        step_size = (ub - lb) / 10
        learning_rate = 0.1

        for _ in range(self.budget - 1):
            mutation_factor = 0.4 + 0.5 * np.random.rand() * self.success_rate  # Change: Learning-based mutation factor
            diff_vector = np.random.randn(self.dim) * mutation_factor * step_size  # Change: Gaussian distributed diff_vector
            candidate_solution = current_solution + diff_vector
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
                step_size *= 1 + learning_rate  # Change: Learning rate affects step size increment
                self.success_rate = min(1.0, self.success_rate + 0.05)
            else:
                step_size *= 0.9  # Change: More conservative decrease in step size
                self.success_rate = max(0.0, self.success_rate - 0.05)

            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

        return self.best_solution, self.best_value