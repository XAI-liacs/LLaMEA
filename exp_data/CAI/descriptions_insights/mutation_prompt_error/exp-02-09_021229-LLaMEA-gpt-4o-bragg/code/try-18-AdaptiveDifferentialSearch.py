import numpy as np

class AdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.success_rate = 0.5
        self.population_size = 5  # Change 1
        self.population = np.random.uniform(-1, 1, (self.population_size, dim))  # Change 2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))  # Change 3
        step_size = (ub - lb) / 10

        for _ in range(self.budget - self.population_size):  # Change 4
            for i in range(self.population_size):  # Change 5
                diff_vector = np.random.uniform(-step_size, step_size, self.dim)
                rand_solution = np.random.uniform(lb, ub, self.dim)
                candidate_solution = self.population[i] + diff_vector + self.success_rate * (rand_solution - self.population[i])  # Change 6
                candidate_solution = np.clip(candidate_solution, lb, ub)
                candidate_value = func(candidate_solution)

                if candidate_value < func(self.population[i]):  # Change 7
                    self.population[i] = candidate_solution
                    step_size *= 1.1  # Change 8
                else:
                    step_size *= 0.8  # Change 9

                if candidate_value < self.best_value:
                    self.best_solution = candidate_solution
                    self.best_value = candidate_value

            self.success_rate = min(1.0, self.success_rate + 0.05)  # Change 10

        return self.best_solution, self.best_value