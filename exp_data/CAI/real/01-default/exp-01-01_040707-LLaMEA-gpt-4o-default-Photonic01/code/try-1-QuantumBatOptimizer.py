import numpy as np

class QuantumBatOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.A = 0.5  # Loudness
        self.r = 0.5  # Pulse rate
        self.freq_min = 0.0
        self.freq_max = 2.0
        self.position = None
        self.velocity = None
        self.fitness = None
        self.best = None
        self.best_fitness = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.zeros((self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.fitness[i]:
                self.fitness[i] = scores[i]
            if scores[i] < self.best_fitness:
                self.best_fitness = scores[i]
                self.best = self.position[i]
        return scores

    def update_velocity_position(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        for i in range(self.population_size):
            freq = self.freq_min + (self.freq_max - self.freq_min) * np.random.rand()
            self.velocity[i] += (self.position[i] - self.best) * freq
            self.position[i] += self.velocity[i]
            self.position[i] = np.clip(self.position[i], lb, ub)
            if np.random.rand() > self.r:
                self.position[i] = self.best + 0.001 * np.random.randn(self.dim)

    def quantum_tunneling(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        for i in range(self.population_size):
            if np.random.rand() < self.r:
                q_position = (1 - self.A) * self.position[i] + self.A * self.best
                q_position = np.clip(q_position + np.random.uniform(-0.5, 0.5, self.dim), lb, ub)
                if func(q_position) < self.fitness[i]:
                    self.position[i] = q_position

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position(func.bounds)
            self.quantum_tunneling(func.bounds)
        return self.best, self.best_fitness