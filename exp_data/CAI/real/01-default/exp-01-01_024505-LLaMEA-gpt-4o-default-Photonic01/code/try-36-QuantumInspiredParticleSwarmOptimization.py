import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, min(60, budget // 20))
        self.velocity = np.random.rand(self.swarm_size, self.dim)
        self.position = None
        self.personal_best_position = None
        self.personal_best_fitness = np.full(self.swarm_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.neighborhood_topology = np.random.randint(0, self.swarm_size, size=(self.swarm_size, 3))

    def initialize_swarm(self, lb, ub):
        self.position = lb + (ub - lb) * np.random.rand(self.swarm_size, self.dim)
        self.personal_best_position = self.position.copy()

    def evaluate_swarm(self, func):
        fitness = np.array([func(ind) for ind in self.position])
        better_mask = fitness < self.personal_best_fitness
        self.personal_best_fitness[better_mask] = fitness[better_mask]
        self.personal_best_position[better_mask] = self.position[better_mask]
        
        global_best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[global_best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[global_best_idx]
            self.global_best_position = self.personal_best_position[global_best_idx]

    def update_velocity_and_position(self, lb, ub):
        for i in range(self.swarm_size):
            local_best_position = self.position[self.neighborhood_topology[i]].min(axis=0)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_component = self.cognitive_coefficient * r1 * (self.personal_best_position[i] - self.position[i])
            social_component = self.social_coefficient * r2 * (local_best_position - self.position[i])
            quantized_update = self.inertia_weight * self.velocity[i] + cognitive_component + social_component
            self.velocity[i] = np.random.normal(loc=quantized_update, scale=0.1)
            self.position[i] += self.velocity[i]
            self.position[i] = np.clip(self.position[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_swarm(func)
            evaluations += self.swarm_size

            if evaluations >= self.budget:
                break

            self.update_velocity_and_position(lb, ub)

            # Dynamic adjustment of neighborhood topology
            if evaluations % (self.budget // 10) == 0:
                self.neighborhood_topology = np.random.randint(0, self.swarm_size, size=(self.swarm_size, 3))

        return self.global_best_position, self.global_best_fitness