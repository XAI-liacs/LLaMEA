import numpy as np

class AdaptiveLevyFlightPSODynamicNeighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min = 0.4
        self.w_max = 0.9
        self.alpha = 0.1
        self.beta = 1.5  # Lévy flight parameter
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

    def update_inertia_weight(self, iteration, max_iterations):
        return self.w_max - ((self.w_max - self.w_min) * (iteration / max_iterations))

    def levy_flight(self, position):
        step = np.random.normal(size=position.shape) * (1 / (np.abs(np.random.normal(scale=1, size=position.shape)) ** (1 / self.beta)))
        return position + self.alpha * step

    def update_velocity_position(self, iteration, max_iterations):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)

        # Dynamic neighborhood selection
        neighborhood_size = min(5, self.population_size // 5)
        neighbors = np.random.choice(self.population_size, size=(self.population_size, neighborhood_size), replace=False)
        neighborhood_best = np.array([self.pbest[neigh].min(axis=0) for neigh in neighbors])

        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (neighborhood_best - self.position)

        self.velocity = inertia_weight * self.velocity + cognitive + social
        self.position += self.velocity

        # Apply Lévy flight for enhanced exploration
        self.position = self.levy_flight(self.position)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position(iteration, max_iterations)
            iteration += 1

        return self.gbest, self.gbest_score