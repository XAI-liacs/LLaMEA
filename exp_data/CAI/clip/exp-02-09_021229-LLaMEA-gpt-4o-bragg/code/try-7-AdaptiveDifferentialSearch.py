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

        for _ in range(self.budget - 1):
            diff_vector = np.random.uniform(-step_size, step_size, self.dim)
            rand_solution = np.random.uniform(lb, ub, self.dim)
            mutation_factor = np.random.uniform(0.4, 0.9)  # Dynamic mutation scaling
            candidate_solution = current_solution + diff_vector + mutation_factor * (rand_solution - current_solution)
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

# Example usage:
# optimizer = AdaptiveDifferentialSearch(budget=100, dim=5)
# best_solution, best_value = optimizer(your_func)