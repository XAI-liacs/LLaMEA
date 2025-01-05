import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.alpha = np.random.rand(self.population_size, self.dim)
        self.beta = np.sqrt(1 - np.square(self.alpha))
        self.rotation_angle = 0.01

    def observe(self):
        return np.where(np.random.rand(self.population_size, self.dim) < np.square(self.alpha), 1, -1)

    def rotate(self, solutions, lb, ub, evaluations):
        for i in range(self.population_size):
            for j in range(self.dim):
                if evaluations < self.budget * 0.5:
                    theta = np.random.uniform(-self.rotation_angle, self.rotation_angle)
                else:
                    theta = self.rotation_angle * (self.best_solution[j] - solutions[i][j])
                self.alpha[i, j] = self.alpha[i, j] * np.cos(theta) - self.beta[i, j] * np.sin(theta)
                self.beta[i, j] = np.sqrt(1 - np.square(self.alpha[i, j]))

            solutions[i] = np.clip(solutions[i], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        solutions = self.observe() * (ub - lb) / 2 + (ub + lb) / 2

        while evaluations < self.budget:
            for i in range(self.population_size):
                solution = solutions[i]
                value = func(solution)
                evaluations += 1
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = solution

                if evaluations >= self.budget:
                    break

            self.rotate(solutions, lb, ub, evaluations)
            solutions = self.observe() * (ub - lb) / 2 + (ub + lb) / 2

        return self.best_solution, self.best_value