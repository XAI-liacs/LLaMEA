import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.crossover_rate = 0.8
        self.learning_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 5  # New population size for diversity
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        values = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        best_idx = np.argmin(values)
        best_solution, best_value = population[best_idx], values[best_idx]

        while self.evaluations < self.budget:
            diversity = np.std(population, axis=0).mean()  # Measure diversity
            self.crossover_rate = 0.5 + 0.5 * (1 - diversity)  # Adapt crossover rate

            for i in range(population_size):
                if np.random.rand() < self.crossover_rate:
                    mutation_scale = np.clip(0.05 + 0.25 * diversity, 0.05, 0.3)
                    candidate = np.random.normal(population[i], (ub - lb) / 10 * mutation_scale, self.dim)
                    candidate = np.clip(candidate, lb, ub)
                else:
                    gradient = self.estimate_gradient(func, population[i], lb, ub)
                    candidate = self.line_search(func, population[i], gradient, lb, ub)

                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < values[i]:
                    population[i], values[i] = candidate, candidate_value
                    if candidate_value < best_value:
                        best_solution, best_value = candidate, candidate_value
                        self.learning_rate *= 1.1 if candidate_value < best_value else 1.05

        return best_solution, best_value

    def estimate_gradient(self, func, solution, lb, ub, epsilon=1e-4):
        gradient = np.zeros(self.dim)
        adaptive_epsilon = epsilon * (1 + self.evaluations / self.budget)
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