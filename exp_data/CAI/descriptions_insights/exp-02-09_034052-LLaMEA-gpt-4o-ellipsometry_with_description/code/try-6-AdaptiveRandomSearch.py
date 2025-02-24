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
        restart_interval = 10
        annealing_rate = 0.95

        while evaluations < self.budget:
            if evaluations % restart_interval == 0:
                current_solution = np.random.uniform(lb, ub, self.dim)
                step_size = (ub - lb) / 10

            current_score = func(current_solution)
            evaluations += 1

            candidate_solution = current_solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_score = func(candidate_solution)
            evaluations += 1

            acceptance_probability = np.exp((current_score - candidate_score) / step_size)  # New line for stochastic acceptance
            if candidate_score < current_score or np.random.rand() < acceptance_probability:  # Changed line for acceptance criteria
                current_solution = candidate_solution
                current_score = candidate_score
                step_size *= annealing_rate
                restart_interval = max(5, restart_interval - 1)
            else:
                step_size *= 1.15

            if current_score < best_score:
                best_solution = current_solution
                best_score = current_score

        return best_solution