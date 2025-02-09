import numpy as np

class AdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_solution)
        step_size = (ub - lb) / 10

        for iteration in range(self.budget - 1):
            oscillation_factor = 1.0 + 0.5 * np.sin(2 * np.pi * iteration / (self.budget))  # Change 1
            diff_vector = np.random.uniform(-step_size, step_size, self.dim) * (0.5 + 0.5 * np.random.rand()) * oscillation_factor
            rand_solution = np.random.uniform(lb, ub, self.dim)
            candidate_solution = current_solution + diff_vector + 0.5 * (rand_solution - current_solution)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
                step_size *= 1.2
            else:
                step_size *= 0.9

            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

        return self.best_solution, self.best_value