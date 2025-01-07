import numpy as np

class HierarchicalCoEvolutionaryParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.sub_population_size = 10
        self.c1 = 1.5
        self.c2 = 2.0
        self.w_min = 0.4
        self.w_max = 0.9
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

    def update_velocity_position(self, iteration, max_iterations):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive + social
        self.position += self.velocity

    def dynamic_leader_selection(self):
        leaders = []
        for i in range(0, self.population_size, self.sub_population_size):
            sub_positions = self.position[i:i+self.sub_population_size]
            sub_scores = self.pbest_scores[i:i+self.sub_population_size]
            leader_idx = np.argmin(sub_scores)
            leaders.append(sub_positions[leader_idx])
        return np.array(leaders)

    def neighborhood_learning(self, leaders):
        for i in range(self.population_size):
            neighbor_idx = np.random.choice(range(leaders.shape[0]))
            self.position[i] = self.position[i] + np.random.rand() * (leaders[neighbor_idx] - self.position[i])

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position(iteration, max_iterations)
            leaders = self.dynamic_leader_selection()
            self.neighborhood_learning(leaders)
            iteration += 1
        return self.gbest, self.gbest_score