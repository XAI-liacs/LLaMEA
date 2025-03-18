import numpy as np

class AdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_temperature = 1.0
        self.final_temperature = 1e-3
        self.alpha = 0.9  # Cooling rate

    def __call__(self, func):
        num_evaluations = 0
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub

        # Initialize solution
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_score = func(current_solution)
        num_evaluations += 1
        best_solution = np.copy(current_solution)
        best_score = current_score

        temperature = self.initial_temperature

        while num_evaluations < self.budget and temperature > self.final_temperature:
            # Generate a new candidate solution
            candidate_solution = current_solution + np.random.uniform(-temperature, temperature, self.dim)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_score = func(candidate_solution)
            num_evaluations += 1

            # Decide whether to accept the candidate solution
            acceptance_probability = np.exp((current_score - candidate_score) / temperature)
            if candidate_score < current_score or np.random.rand() < acceptance_probability:
                current_solution = candidate_solution
                current_score = candidate_score

            # Update the best solution found so far
            if current_score < best_score:
                best_solution = np.copy(current_solution)
                best_score = current_score

            # Reduce the temperature
            temperature *= self.alpha

        return best_solution