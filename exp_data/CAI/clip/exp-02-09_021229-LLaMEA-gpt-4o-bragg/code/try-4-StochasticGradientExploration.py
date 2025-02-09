import numpy as np

class StochasticGradientExploration:
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

        for _ in range(self.budget - 1):
            noise = np.random.normal(size=self.dim)
            candidate_solution = current_solution + step_size * noise
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                gradient_estimate = (candidate_value - current_value) / (step_size * np.linalg.norm(noise))
                current_solution -= step_size * gradient_estimate * noise
                current_value = func(current_solution)
                step_size *= 1.1
            else:
                step_size *= 0.9

            if current_value < self.best_value:
                self.best_solution = current_solution
                self.best_value = current_value

        return self.best_solution, self.best_value

# Example usage:
# optimizer = StochasticGradientExploration(budget=100, dim=5)
# best_solution, best_value = optimizer(your_func)