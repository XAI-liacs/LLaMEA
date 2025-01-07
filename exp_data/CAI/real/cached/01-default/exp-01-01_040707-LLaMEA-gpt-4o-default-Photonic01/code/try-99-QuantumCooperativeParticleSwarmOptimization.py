import numpy as np

class QuantumCooperativeParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.inertia_weight = 0.7
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

    def quantum_position_update(self, current_position, best_position):
        phi = np.random.rand(*current_position.shape)
        return current_position + phi * (best_position - current_position)

    def mutate(self, position, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        mutation = np.where(np.random.rand(self.dim) < self.mutation_rate,
                            lb + (ub - lb) * np.random.rand(self.dim), position)
        return mutation

    def update_velocity(self):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_coefficient * r1 * (self.pbest[i] - self.position[i])
            social_velocity = self.social_coefficient * r2 * (self.gbest - self.position[i])
            self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                cognitive_velocity +
                                social_velocity)

    def update_position(self, bounds):
        self.position += self.velocity
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = np.clip(self.position, lb, ub)
        for i in range(self.population_size):
            self.position[i] = self.mutate(self.position[i], bounds)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity()
            self.update_position(func.bounds)
            self.position = self.quantum_position_update(self.position, self.gbest)

        return self.gbest, self.gbest_score