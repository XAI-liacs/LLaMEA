import numpy as np

class ALF_GHO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.positions = None
        self.best_position = None
        self.best_value = np.inf
        self.history = []  # Store historical best values
        self.alpha = 0.1   # Step size for gradient-based local search
        self.beta = 1.5    # Lévy distribution exponent
        self.sigma = (np.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                      (np.gamma((1 + self.beta) / 2) * self.beta * np.power(2, (self.beta - 1) / 2)))**(1 / self.beta)

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.bounds = (lb, ub)

    def levy_flight(self, position, best_position):
        u = np.random.normal(0, self.sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1/self.beta)
        new_position = position + step * (position - best_position)
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def gradient_local_search(self, func, position):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            epsilon = 1e-8
            step = np.zeros(self.dim)
            step[i] = epsilon
            grad[i] = (func(position + step) - func(position - step)) / (2 * epsilon)
        new_position = position - self.alpha * grad
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.positions[i])
                evaluations += 1

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_position = self.positions[i].copy()

            self.history.append(self.best_value)

            for i in range(self.population_size):
                self.positions[i] = self.levy_flight(self.positions[i], self.best_position)

                if np.random.rand() < 0.2:  # Perform local search with probability 0.2
                    self.positions[i] = self.gradient_local_search(func, self.positions[i])

        return self.best_position, self.best_value