import numpy as np

class EnhancedDynamicAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.history = []  # To store past solutions for diversity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        step_size = (ub - lb) / 10
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_solution)

        for _ in range(self.budget - 1):
            # Introduce diversity by perturbing the step size occasionally
            if np.random.rand() < 0.1:
                step_size = step_size * np.random.uniform(0.8, 1.2)
            
            candidate_solution = current_solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
                step_size *= 1.1  # Less aggressive increase
            else:
                step_size *= 0.8  # More aggressive decrease
                # Adaptively learn from history
                if len(self.history) > 0:
                    past_solution = self.history[np.random.randint(0, len(self.history))]
                    candidate_solution = past_solution + np.random.uniform(-step_size, step_size, self.dim)
                    candidate_solution = np.clip(candidate_solution, lb, ub)
                    candidate_value = func(candidate_solution)
                    if candidate_value < current_value:
                        current_solution = candidate_solution
                        current_value = candidate_value

            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

            self.history.append(current_solution)

        return self.best_solution, self.best_value