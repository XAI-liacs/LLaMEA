import numpy as np

class GE_AMSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.memory_archive = []
        self.archive_size = 20
        self.population_size = 10
        self.gradient_steps = 5

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def update_memory_archive(self, solution, value):
        self.memory_archive.append((solution.copy(), value))
        self.memory_archive = sorted(self.memory_archive, key=lambda x: x[1])[:self.archive_size]

    def adaptive_exploration(self, lb, ub):
        if self.memory_archive:
            selected_solution, _ = self.memory_archive[np.random.randint(len(self.memory_archive))]
            perturbation = np.random.normal(0, 0.1, self.dim)
            candidate = selected_solution + perturbation * (ub - lb)
            return np.clip(candidate, lb, ub)
        else:
            return lb + (ub - lb) * np.random.rand(self.dim)

    def local_refinement(self, solution, lb, ub):
        perturbation = np.random.normal(0, 0.01, self.dim)
        new_solution = solution + perturbation * (ub - lb)
        return np.clip(new_solution, lb, ub)

    def estimate_gradient(self, solution, func, lb, ub):
        grad = np.zeros(self.dim)
        sigma = 0.01 * (ub - lb)
        for i in range(self.dim):
            step = np.zeros(self.dim)
            step[i] = sigma[i]
            value1 = func(solution + step)
            value2 = func(solution - step)
            grad[i] = (value1 - value2) / (2 * sigma[i])
        return grad

    def gradient_refinement(self, solution, func, lb, ub):
        current_solution = solution.copy()
        for _ in range(self.gradient_steps):
            grad = self.estimate_gradient(current_solution, func, lb, ub)
            current_solution -= 0.1 * grad * (ub - lb)
            current_solution = np.clip(current_solution, lb, ub)
        return current_solution

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            candidate_solutions = []
            for _ in range(self.population_size):
                if evaluations < self.budget // 3:
                    candidate = self.adaptive_exploration(lb, ub)
                elif evaluations < 2 * self.budget // 3:
                    candidate = self.local_refinement(self.best_solution, lb, ub)
                else:
                    candidate = self.gradient_refinement(self.best_solution, func, lb, ub)
                candidate_solutions.append(candidate)

            for solution in candidate_solutions:
                value = func(solution)
                evaluations += 1
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = solution
                self.update_memory_archive(solution, value)

                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value