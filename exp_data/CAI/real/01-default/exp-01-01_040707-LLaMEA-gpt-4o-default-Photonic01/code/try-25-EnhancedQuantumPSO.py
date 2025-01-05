import numpy as np

class EnhancedQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_min = 0.2
        self.w_max = 0.8
        self.rotation_angle = np.pi / 4
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.neighborhood_size = max(2, self.population_size // 5)

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

    def quantum_rotation_gate(self, velocity):
        theta = np.random.rand(*velocity.shape) * self.rotation_angle
        q_velocity = velocity * np.cos(theta) + np.random.rand(*velocity.shape) * np.sin(theta)
        return q_velocity

    def dynamic_neighborhood_best(self, index):
        start = max(0, index - self.neighborhood_size // 2)
        end = min(self.population_size, index + self.neighborhood_size // 2)
        neighborhood = self.position[start:end]
        neighborhood_scores = self.pbest_scores[start:end]
        local_best_index = np.argmin(neighborhood_scores)
        return neighborhood[local_best_index]

    def update_velocity_position(self, iteration, max_iterations):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)
        for i in range(self.population_size):
            cognitive = self.c1 * r1[i] * (self.pbest[i] - self.position[i])
            local_best = self.dynamic_neighborhood_best(i)
            social = self.c2 * r2[i] * (local_best - self.position[i])
            quantum_velocity = self.quantum_rotation_gate(self.velocity[i])
            self.velocity[i] = inertia_weight * quantum_velocity + cognitive + social
            self.position[i] += self.velocity[i]

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