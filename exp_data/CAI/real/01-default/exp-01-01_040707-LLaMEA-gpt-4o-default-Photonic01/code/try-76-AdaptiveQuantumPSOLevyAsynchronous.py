import numpy as np

class AdaptiveQuantumPSOLevyAsynchronous:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_min = 0.2
        self.w_max = 0.8
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.func_calls = 0

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        self.func_calls += self.population_size
        return scores

    def update_inertia_weight(self, iteration, max_iterations):
        return self.w_max - ((self.w_max - self.w_min) * (iteration / max_iterations))

    def quantum_rotation_gate(self, velocity):
        rotation_angle = np.pi / 4
        theta = np.random.rand(*velocity.shape) * rotation_angle
        q_velocity = velocity * np.cos(theta) + np.random.rand(*velocity.shape) * np.sin(theta)
        return q_velocity

    def levy_flight(self, position):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(*position.shape) * sigma
        v = np.random.randn(*position.shape)
        step = u / abs(v) ** (1 / beta)
        return position + 0.01 * step

    def update_velocity_position(self, iteration, max_iterations):
        for i in range(self.population_size):
            if self.func_calls >= self.budget:
                break
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            inertia_weight = self.update_inertia_weight(iteration, max_iterations)
            cognitive = self.c1 * r1 * (self.pbest[i] - self.position[i])
            social = self.c2 * r2 * (self.gbest - self.position[i])

            quantum_velocity = self.quantum_rotation_gate(self.velocity[i])
            self.velocity[i] = inertia_weight * quantum_velocity + cognitive + social
            self.position[i] += self.velocity[i]
            self.position[i] = self.levy_flight(self.position[i])

    def __call__(self, func):
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while self.func_calls < self.budget:
            self.evaluate(func)
            self.update_velocity_position(iteration, max_iterations)
            iteration += 1

        return self.gbest, self.gbest_score