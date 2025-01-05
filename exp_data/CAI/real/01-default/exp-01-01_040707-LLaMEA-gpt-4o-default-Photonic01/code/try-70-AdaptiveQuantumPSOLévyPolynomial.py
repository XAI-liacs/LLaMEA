import numpy as np

class AdaptiveQuantumPSOLévyPolynomial:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5  # Fixed inertia weight to simplify tuning
        self.rotation_angle = np.pi / 4
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

    def quantum_rotation_gate(self, velocity):
        theta = np.random.rand(*velocity.shape) * self.rotation_angle
        q_velocity = velocity * np.cos(theta) + np.random.rand(*velocity.shape) * np.sin(theta)
        return q_velocity

    def lévy_flight(self, position):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, position.shape)
        v = np.random.normal(0, 1, position.shape)
        step = u / np.abs(v)**(1 / beta)
        return position + step

    def polynomial_mutation(self, position):
        eta_m = 20.0
        delta = np.random.rand(*position.shape)
        for i in range(position.shape[0]):
            for j in range(position.shape[1]):
                if delta[i, j] < 0.5:
                    delta[i, j] = (2.0 * delta[i, j])**(1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta[i, j] = 1.0 - (2.0 * (1.0 - delta[i, j]))**(1.0 / (eta_m + 1.0))
        return position + delta

    def update_velocity_position(self):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)
        quantum_velocity = self.quantum_rotation_gate(self.velocity)
        self.velocity = self.w * quantum_velocity + cognitive + social
        self.position += self.velocity
        self.position = self.lévy_flight(self.position)
        self.position = self.polynomial_mutation(self.position)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position()

        return self.gbest, self.gbest_score