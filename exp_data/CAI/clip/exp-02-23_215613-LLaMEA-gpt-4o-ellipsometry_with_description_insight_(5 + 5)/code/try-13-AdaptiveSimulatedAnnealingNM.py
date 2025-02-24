import numpy as np
from scipy.optimize import minimize

class AdaptiveSimulatedAnnealingNM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_temp = 1.0
        self.cooling_rate = 0.95

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        current_point = np.random.uniform(bounds[:, 0], bounds[:, 1])
        current_value = func(current_point)
        best_point = current_point
        best_value = current_value
        temperature = self.initial_temp

        while self.evaluations < self.budget:
            # Generate a new candidate solution
            candidate_point = current_point + np.random.uniform(-0.1, 0.1, self.dim) * temperature
            candidate_point = np.clip(candidate_point, bounds[:, 0], bounds[:, 1])
            candidate_value = func(candidate_point)
            self.evaluations += 1

            # Decide whether to accept the new candidate
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / temperature):
                current_point, current_value = candidate_point, candidate_value

            # Apply local search to refine the candidate
            if self.evaluations < self.budget:
                result = minimize(func, current_point, method='Nelder-Mead', options={'maxfev': min(self.budget - self.evaluations, 50)})
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                self.evaluations += result.nfev

            # Update temperature
            temperature *= self.cooling_rate

        return best_point