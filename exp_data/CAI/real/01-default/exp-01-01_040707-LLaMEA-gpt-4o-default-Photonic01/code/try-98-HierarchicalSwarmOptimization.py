import numpy as np

class HierarchicalSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5  # weight for global and local influence
        self.beta = 0.8   # inertia weight
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.scale_factors = np.logspace(-2, 0, num=3)  # multi-scale exploration factors

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) * 0.1 * (ub - lb)
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

    def update_velocity_and_position(self, scale_factor):
        r1, r2 = np.random.rand(2)
        for i in range(self.population_size):
            local_influence = r1 * (self.pbest[i] - self.position[i])
            global_influence = r2 * (self.gbest - self.position[i])
            self.velocity[i] = (self.beta * self.velocity[i] +
                                self.alpha * (local_influence + global_influence) * scale_factor)
            self.position[i] += self.velocity[i]

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            for scale_factor in self.scale_factors:
                self.update_velocity_and_position(scale_factor)
                self.position = np.clip(self.position, func.bounds.lb, func.bounds.ub)

        return self.gbest, self.gbest_score