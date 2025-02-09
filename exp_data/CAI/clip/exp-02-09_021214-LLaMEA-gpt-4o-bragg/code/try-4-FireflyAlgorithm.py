import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_fireflies = 10
        self.alpha = 0.2  # randomization parameter
        self.beta0 = 1.0  # base attractiveness
        self.gamma = 1.0  # absorption coefficient
        self.best_solution = None
        self.best_obj = float('inf')

    def initialize_fireflies(self, bounds):
        self.lb, self.ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_fireflies, self.dim))
        self.intensities = np.array([float('inf')] * self.num_fireflies)

    def attractiveness(self, distance):
        return self.beta0 * np.exp(-self.gamma * distance ** 2)

    def update_positions(self):
        for i in range(self.num_fireflies):
            for j in range(self.num_fireflies):
                if self.intensities[i] > self.intensities[j]:  # Move firefly i towards j
                    distance = np.linalg.norm(self.positions[i] - self.positions[j])
                    beta = self.attractiveness(distance)
                    self.positions[i] += beta * (self.positions[j] - self.positions[i]) \
                                         + self.alpha * (np.random.rand(self.dim) - 0.5)
                    self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

    def __call__(self, func):
        self.initialize_fireflies(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            objectives = [func(pos) for pos in self.positions]
            evaluations += self.num_fireflies

            for i in range(self.num_fireflies):
                self.intensities[i] = objectives[i]
                if objectives[i] < self.best_obj:
                    self.best_obj = objectives[i]
                    self.best_solution = self.positions[i]

            self.update_positions()
        
        return self.best_solution