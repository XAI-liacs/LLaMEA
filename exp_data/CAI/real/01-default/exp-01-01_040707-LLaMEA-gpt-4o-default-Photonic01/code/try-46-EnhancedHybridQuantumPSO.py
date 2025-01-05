import numpy as np

class EnhancedHybridQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_min = 0.2
        self.w_max = 0.8
        self.mutation_rate_min = 0.05
        self.mutation_rate_max = 0.3
        self.rotation_angle_max = np.pi / 4
        self.rotation_angle_min = np.pi / 8
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
        return self.w_max - ((self.w_max - self.w_min) * iteration / max_iterations)

    def adaptive_quantum_rotation(self, velocity, iteration, max_iterations):
        rotation_angle = self.rotation_angle_max - ((self.rotation_angle_max - self.rotation_angle_min) * iteration / max_iterations)
        theta = np.random.rand(*velocity.shape) * rotation_angle
        q_velocity = velocity * np.cos(theta) + np.random.rand(*velocity.shape) * np.sin(theta)
        return q_velocity

    def adaptive_gaussian_mutation(self, position, iteration, max_iterations):
        mutation_rate = self.mutation_rate_min + ((self.mutation_rate_max - self.mutation_rate_min) * iteration / max_iterations)
        mutation = np.random.randn(*position.shape) * mutation_rate
        return position + mutation

    def update_velocity_position(self, iteration, max_iterations):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)

        quantum_velocity = self.adaptive_quantum_rotation(self.velocity, iteration, max_iterations)
        self.velocity = inertia_weight * quantum_velocity + cognitive + social
        self.position += self.velocity
        self.position = self.adaptive_gaussian_mutation(self.position, iteration, max_iterations)

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