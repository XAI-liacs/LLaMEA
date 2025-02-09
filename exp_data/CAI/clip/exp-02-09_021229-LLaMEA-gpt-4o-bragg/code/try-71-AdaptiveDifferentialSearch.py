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

        for iter_count in range(self.budget - 1):
            mutation_factor = np.random.uniform(0.5, 1.0) * (1 - iter_count / self.budget)  # Change 1
            diff_vector = np.random.uniform(-step_size, step_size, self.dim) * mutation_factor
            rand_solution = np.random.uniform(lb, ub, self.dim)
            candidate_solution = current_solution + diff_vector + self.success_rate * (rand_solution - current_solution)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
                step_size *= 1.3  # Change 2
                self.success_rate = min(1.0, self.success_rate + 0.06)  # Change 3
            else:
                step_size *= 0.8 + 0.2 * np.random.rand()  # Change 4
                self.success_rate = max(0.0, self.success_rate - 0.04)  # Change 5

            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

        return self.best_solution, self.best_value