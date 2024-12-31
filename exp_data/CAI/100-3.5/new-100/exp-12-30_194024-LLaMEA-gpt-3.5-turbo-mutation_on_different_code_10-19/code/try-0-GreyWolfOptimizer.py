import numpy as np

class GreyWolfOptimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 5
        self.positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.alpha, self.beta, self.delta = None, None, None

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                if self.fitness[i] < np.inf:
                    continue
                fitness_val = func(self.positions[i])
                if fitness_val < self.fitness[i]:
                    self.fitness[i] = fitness_val

            alpha_index = np.argmin(self.fitness)
            self.alpha = self.positions[alpha_index]

            for i in range(self.population_size):
                a = 2 - 2 * (_ / self.budget)  # Decreasing encircling behavior
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha - self.positions[i])
                X1 = self.alpha - A1 * D_alpha

                beta_index = np.argsort(self.fitness)[1]
                self.beta = self.positions[beta_index]
                D_beta = np.abs(C1 * self.beta - self.positions[i])
                X2 = self.beta - A1 * D_beta

                delta_index = np.argsort(self.fitness)[2]
                self.delta = self.positions[delta_index]
                D_delta = np.abs(C1 * self.delta - self.positions[i])
                X3 = self.delta - A1 * D_delta

                self.positions[i] = (X1 + X2 + X3) / 3

        best_index = np.argmin(self.fitness)
        return self.fitness[best_index], self.positions[best_index]