import numpy as np

class LevyFlightEnhancedPSOSGD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.alpha = 0.01  # Learning rate for SGD
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        return scores

    def levy_flight(self, step_size=0.1):
        u = np.random.randn(*self.position.shape) * step_size
        v = np.random.randn(*self.position.shape)
        step = u / (np.abs(v) ** (1 / 3))
        return self.position + step

    def sgd_update(self, func):
        gradients = np.array([self.gradient_approx(func, p) for p in self.position])
        self.position -= self.alpha * gradients

    def gradient_approx(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] += epsilon
            x2[i] -= epsilon
            grad[i] = (func(x1) - func(x2)) / (2 * epsilon)
        return grad

    def update_velocity_position(self):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)
        self.velocity = self.w * self.velocity + cognitive + social
        self.position += self.velocity

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            if iteration % 2 == 0:
                self.position = self.levy_flight()
            self.update_velocity_position()
            self.sgd_update(func)
            iteration += 1

        return self.gbest, self.gbest_score