import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, self.dim)
        best_value = func(best_solution)
        self.evaluations += 1
        
        while self.evaluations < self.budget:
            if np.random.rand() < 0.5:
                # Gradient-free exploration
                candidate = np.random.uniform(lb, ub, self.dim)
            else:
                # Gradient-based exploitation
                gradient = self.estimate_gradient(func, best_solution, lb, ub)
                candidate = self.line_search(func, best_solution, gradient, lb, ub)

            candidate_value = func(candidate)
            self.evaluations += 1
            
            if candidate_value < best_value:
                best_solution, best_value = candidate, candidate_value

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
        step_size = alpha
        while step_size > 1e-5:
            candidate = solution - step_size * gradient
            candidate = np.clip(candidate, lb, ub)
            if func(candidate) < func(solution):
                return candidate
            step_size *= beta
            self.evaluations += 1
        return solution