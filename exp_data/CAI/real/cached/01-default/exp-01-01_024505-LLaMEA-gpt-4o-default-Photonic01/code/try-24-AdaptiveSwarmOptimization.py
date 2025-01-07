import numpy as np

class AdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, min(60, budget // 5))
        self.position = None
        self.velocity = None
        self.best_swarm_position = None
        self.best_swarm_fitness = float('inf')
        self.individual_best_positions = None
        self.individual_best_fitness = None
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5

    def initialize_swarm(self, lb, ub):
        self.position = lb + (ub - lb) * np.random.rand(self.swarm_size, self.dim)
        self.velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.individual_best_positions = np.copy(self.position)
        self.individual_best_fitness = np.full(self.swarm_size, float('inf'))

    def evaluate_swarm(self, func):
        fitness = np.array([func(ind) for ind in self.position])
        for i in range(self.swarm_size):
            if fitness[i] < self.individual_best_fitness[i]:
                self.individual_best_fitness[i] = fitness[i]
                self.individual_best_positions[i] = self.position[i]
            if fitness[i] < self.best_swarm_fitness:
                self.best_swarm_fitness = fitness[i]
                self.best_swarm_position = self.position[i]
        return fitness

    def update_velocity_and_position(self, lb, ub):
        r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
        cognitive_component = self.cognitive_coefficient * r1 * (self.individual_best_positions - self.position)
        social_component = self.social_coefficient * r2 * (self.best_swarm_position - self.position)
        self.velocity = (self.inertia_weight * self.velocity + cognitive_component + social_component)
        self.position += self.velocity
        self.position = np.clip(self.position, lb, ub)

    def adapt_parameters(self, evaluations):
        self.inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
        self.cognitive_coefficient = 1.5 + (evaluations / self.budget)
        self.social_coefficient = 1.5 + (1 - evaluations / self.budget)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_swarm(func)
            evaluations += self.swarm_size

            if evaluations >= self.budget:
                break

            self.update_velocity_and_position(lb, ub)
            self.adapt_parameters(evaluations)

        return self.best_swarm_position, self.best_swarm_fitness