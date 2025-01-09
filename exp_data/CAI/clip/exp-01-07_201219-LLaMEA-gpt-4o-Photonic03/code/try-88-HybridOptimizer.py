import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.crossover_rate = 0.8
        self.learning_rate = 0.1  # Initial learning rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, self.dim)
        best_value = func(best_solution)
        self.evaluations += 1
        previous_value = best_value  # Track previous iteration's value

        while self.evaluations < self.budget:
            if np.random.rand() < self.crossover_rate:
                # Adaptive mutation factor based on evaluation progress
                mutation_scale = np.clip(0.1 + 0.4 * np.sin(self.evaluations / self.budget * np.pi), 0.05, 0.35)
                candidate = np.random.normal(best_solution, (ub - lb) / 10 * mutation_scale, self.dim)
                candidate = np.clip(candidate, lb, ub)
            else:
                gradient = self.estimate_gradient(func, best_solution, lb, ub)
                candidate = self.line_search(func, best_solution, gradient, lb, ub)

            candidate_value = func(candidate)
            self.evaluations += 1

            if candidate_value < best_value:
                best_solution, best_value = candidate, candidate_value
                self.crossover_rate = min(1.0, self.crossover_rate + 0.02 * np.exp(-self.evaluations / self.budget))
                # More adaptable learning rate scaling
                self.learning_rate *= 1.1 if candidate_value < previous_value else 1.05
                previous_value = candidate_value
            else:
                self.crossover_rate = max(0.2, self.crossover_rate - 0.01 * np.random.rand())
                self.learning_rate *= 0.93

        return best_solution, best_value

    def estimate_gradient(self, func, solution, lb, ub, epsilon=1e-4):
        gradient = np.zeros(self.dim)
        adaptive_epsilon = epsilon * (1 + self.evaluations / self.budget)  # Adaptive epsilon
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = adaptive_epsilon
            upper_sol = np.clip(solution + perturb, lb, ub)
            lower_sol = np.clip(solution - perturb, lb, ub)
            gradient[i] = (func(upper_sol) - func(lower_sol)) / adaptive_epsilon
            self.evaluations += 2
        return gradient

    def line_search(self, func, solution, gradient, lb, ub, alpha=1e-2, beta=0.9):
        step_size = self.learning_rate * np.linalg.norm(gradient) / (1 + self.evaluations / self.budget)
        while step_size > 1e-5:
            candidate = solution - step_size * gradient
            candidate = np.clip(candidate, lb, ub)
            if func(candidate) < func(solution):
                return candidate
            step_size *= beta
            self.evaluations += 1
        return solution