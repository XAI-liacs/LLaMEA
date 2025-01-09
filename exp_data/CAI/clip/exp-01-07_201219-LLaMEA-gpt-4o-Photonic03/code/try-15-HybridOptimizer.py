import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.crossover_rate = 0.8  # Adjusted initial adaptive crossover rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, self.dim)
        best_value = func(best_solution)
        self.evaluations += 1

        while self.evaluations < self.budget:
            if np.random.rand() < self.crossover_rate:
                # Gradient-free exploration
                mutation_scale = (0.7 + 0.3 * (self.evaluations / self.budget))  # Adjusted adaptive mutation scale
                candidate = np.random.normal(best_solution, (ub - lb) / 10 * mutation_scale, self.dim)
                candidate = np.clip(candidate, lb, ub)
            else:
                # Gradient-based exploitation
                gradient = self.estimate_gradient(func, best_solution, lb, ub)
                candidate = self.line_search(func, best_solution, gradient, lb, ub)

            candidate_value = func(candidate)
            self.evaluations += 1

            if candidate_value < best_value:
                best_solution, best_value = candidate, candidate_value
                self.crossover_rate = min(1.0, self.crossover_rate + 0.015)  # More dynamic increase
            else:
                self.crossover_rate = max(0.2, self.crossover_rate - 0.01)

        return best_solution, best_value

    def estimate_gradient(self, func, solution, lb, ub, epsilon=1e-4):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = epsilon
            upper_sol = np.clip(solution + perturb, lb, ub)
            lower_sol = np.clip(solution - perturb, lb, ub)
            gradient[i] = (func(upper_sol) - func(lower_sol)) / (2 * epsilon)
            self.evaluations += 2
        return gradient

    def line_search(self, func, solution, gradient, lb, ub, alpha=1e-2, beta=0.9):
        step_size = alpha * np.linalg.norm(gradient) / (1 + self.evaluations / self.budget)
        while step_size > 1e-5:
            candidate = solution - step_size * gradient
            candidate = np.clip(candidate, lb, ub)
            if func(candidate) < func(solution):
                return candidate
            step_size *= beta
            self.evaluations += 1
        return solution