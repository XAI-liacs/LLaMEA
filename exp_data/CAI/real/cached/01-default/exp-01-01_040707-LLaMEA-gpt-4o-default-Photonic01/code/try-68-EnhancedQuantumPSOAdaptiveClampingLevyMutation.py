import numpy as np
from scipy.stats import levy

class EnhancedQuantumPSOAdaptiveClampingLevyMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # Increased cognitive parameter
        self.c2 = 2.0  # Increased social parameter
        self.w_min = 0.2
        self.w_max = 0.9  # Increased max inertia
        self.rotation_angle = np.pi / 4
        self.mutation_rate = 0.1
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

    def adaptive_velocity_clamping(self, velocity, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        return np.clip(velocity, lb - self.position, ub - self.position)

    def quantum_rotation_gate(self, velocity):
        theta = np.random.rand(*velocity.shape) * self.rotation_angle
        q_velocity = velocity * np.cos(theta) + np.random.rand(*velocity.shape) * np.sin(theta)
        return q_velocity

    def levy_flight_mutation(self, position):
        steps = levy.rvs(size=position.shape) * self.mutation_rate
        return position + steps

    def update_velocity_position(self, iteration, max_iterations, bounds):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)

        quantum_velocity = self.quantum_rotation_gate(self.velocity)
        self.velocity = inertia_weight * quantum_velocity + cognitive + social
        self.velocity = self.adaptive_velocity_clamping(self.velocity, bounds)
        self.position += self.velocity
        self.position = self.levy_flight_mutation(self.position)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position(iteration, max_iterations, func.bounds)
            iteration += 1

        return self.gbest, self.gbest_score