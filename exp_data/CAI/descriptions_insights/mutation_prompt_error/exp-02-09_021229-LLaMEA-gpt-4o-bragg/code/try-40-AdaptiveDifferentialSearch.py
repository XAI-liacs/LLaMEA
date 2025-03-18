# Description: Introduced adaptive mutation based on solution diversity to balance exploration and exploitation, improving convergence.
# Code:
import numpy as np

class AdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.success_rate = 0.5
        self.solution_history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_solution)
        step_size = (ub - lb) / 10
        self.solution_history.append(current_solution)

        for _ in range(self.budget - 1):
            diversity_factor = np.std(self.solution_history, axis=0).mean()  # Change
            diff_vector = np.random.uniform(-step_size, step_size, self.dim) * (0.5 + 0.5 * np.random.rand())
            rand_solution = np.random.uniform(lb, ub, self.dim)
            candidate_solution = current_solution + diff_vector + self.success_rate * (rand_solution - current_solution) * diversity_factor  # Change
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
                step_size *= 1.2
                self.success_rate = min(1.0, self.success_rate + 0.05)
            else:
                step_size *= 0.9 + 0.1 * np.random.rand()
                self.success_rate = max(0.0, self.success_rate - 0.05)

            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

            self.solution_history.append(candidate_solution)  # Change

        return self.best_solution, self.best_value