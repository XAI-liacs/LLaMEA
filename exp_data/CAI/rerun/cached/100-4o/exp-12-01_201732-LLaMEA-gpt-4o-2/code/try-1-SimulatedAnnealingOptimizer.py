import numpy as np

class SimulatedAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.temperature = 1.0
        self.cooling_rate = 0.99
    
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        current_solution = best_solution
        current_value = best_value
        evaluations = 1

        while evaluations < self.budget:
            candidate_solution = current_solution + np.random.normal(0, self.temperature, self.dim)
            candidate_solution = np.clip(candidate_solution, self.lower_bound, self.upper_bound)
            candidate_value = func(candidate_solution)
            evaluations += 1

            if candidate_value < best_value or np.random.rand() < np.exp((current_value - candidate_value) / self.temperature):
                current_solution = candidate_solution
                current_value = candidate_value

                if candidate_value < best_value:
                    best_solution = candidate_solution
                    best_value = candidate_value

            self.temperature *= self.cooling_rate

        return best_solution