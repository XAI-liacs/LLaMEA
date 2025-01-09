import numpy as np

class AdaptiveRandomWalkOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_step_size = 0.1
        self.step_size_decay = 0.99

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        current_solution = best_solution
        current_value = best_value
        current_step_size = self.initial_step_size
        evals = 1

        while evals < self.budget:
            direction = np.random.normal(size=self.dim)
            direction /= np.linalg.norm(direction)
            candidate_solution = current_solution + current_step_size * direction
            candidate_solution = np.clip(candidate_solution, self.lower_bound, self.upper_bound)
            candidate_value = func(candidate_solution)

            evals += 1

            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
                current_step_size *= 1.02  # Slightly increase step size on improvement
            else:
                current_step_size *= self.step_size_decay  # Decay step size on no improvement

            if current_value < best_value:
                best_solution = current_solution
                best_value = current_value

        return best_solution, best_value