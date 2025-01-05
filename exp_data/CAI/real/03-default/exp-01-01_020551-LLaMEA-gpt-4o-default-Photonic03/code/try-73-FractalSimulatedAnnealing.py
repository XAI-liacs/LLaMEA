import numpy as np

class FractalSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_temp = 1.0
        self.cooling_rate = 0.99
        self.fractal_perturbation = 0.5
        self.perturbation_decay = 0.95

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_solution = np.random.uniform(lb, ub, self.dim)
        current_score = func(current_solution)
        best_solution = current_solution.copy()
        best_score = current_score
        temperature = self.initial_temp
        evaluations = 1

        while evaluations < self.budget:
            # Fractal perturbation based candidate generation
            perturbation = np.power(self.fractal_perturbation, np.random.normal(size=self.dim))
            candidate_solution = current_solution + perturbation * (ub - lb)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_score = func(candidate_solution)
            evaluations += 1
            
            # Acceptance logic based on simulated annealing
            if candidate_score < current_score or \
               np.random.rand() < np.exp((current_score - candidate_score) / temperature):
                current_solution = candidate_solution
                current_score = candidate_score

                if current_score < best_score:
                    best_solution = current_solution
                    best_score = current_score

            # Update temperature and fractal perturbation
            temperature *= self.cooling_rate
            self.fractal_perturbation *= self.perturbation_decay

        return best_solution, best_score