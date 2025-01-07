import numpy as np

class AdaptiveLevyPSODynamicNeighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_min = 0.4
        self.w_max = 0.9
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.alpha = 1.5  # Lévy distribution exponent

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
        return self.w_max - ((self.w_max - self.w_min) * iteration / max_iterations)

    def levy_flight(self, position):
        step = np.random.standard_cauchy(position.shape) * 0.01
        return position + step

    def update_velocity_position(self, iteration, max_iterations):
        neighbors = np.random.randint(self.population_size, size=(self.population_size, 2))
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)
        
        for i in range(self.population_size):
            local_best = min(self.pbest_scores[neighbors[i]], default=self.gbest_score)
            cognitive = self.c1 * r1[i] * (self.pbest[i] - self.position[i])
            social = self.c2 * r2[i] * (self.gbest - self.position[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive + social

        self.position += self.velocity
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