import numpy as np

class AdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_score = float('inf')
        evaluations = 0
        step_size = (ub - lb) / 10

        # Initial random solution
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_score = func(current_solution)
        evaluations += 1

        while evaluations < self.budget:
            # Generate candidate solution by random sampling within step size bounds
            candidate_solution = current_solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_score = func(candidate_solution)
            evaluations += 1

            # If candidate solution is better, adopt it and reduce step size
            if candidate_score < current_score:
                current_solution = candidate_solution
                current_score = candidate_score
                step_size *= 0.9  # Reduce step size
            else:
                # If no improvement, increase step size
                step_size *= 1.1

            # Update best solution found so far
            if current_score < best_score:
                best_solution = current_solution
                best_score = current_score

        return best_solution