import numpy as np

class AdaptiveQuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget // 10)
        self.particles = None
        self.velocities = None
        self.pbest_pos = None
        self.pbest_fitness = None
        self.gbest_pos = None
        self.gbest_fitness = float('inf')
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_jump_prob = 0.3

    def initialize_swarm(self, lb, ub):
        self.particles = lb + (ub - lb) * np.random.rand(self.swarm_size, self.dim)
        self.velocities = np.zeros_like(self.particles)
        self.pbest_pos = np.copy(self.particles)
        self.pbest_fitness = np.full(self.swarm_size, float('inf'))

    def evaluate_swarm(self, func):
        fitness = np.array([func(p) for p in self.particles])
        for i in range(self.swarm_size):
            if fitness[i] < self.pbest_fitness[i]:
                self.pbest_fitness[i] = fitness[i]
                self.pbest_pos[i] = self.particles[i]
            if fitness[i] < self.gbest_fitness:
                self.gbest_fitness = fitness[i]
                self.gbest_pos = self.particles[i]

    def update_velocities_and_positions(self, lb, ub):
        r1 = np.random.rand(self.swarm_size, self.dim)
        r2 = np.random.rand(self.swarm_size, self.dim)
        cognitive_component = self.cognitive_coeff * r1 * (self.pbest_pos - self.particles)
        social_component = self.social_coeff * r2 * (self.gbest_pos - self.particles)
        self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
        self.particles += self.velocities
        self.particles = np.clip(self.particles, lb, ub)

    def quantum_jump(self, lb, ub):
        for i in range(self.swarm_size):
            if np.random.rand() < self.quantum_jump_prob:
                self.particles[i] = lb + (ub - lb) * np.random.rand(self.dim)

    def adapt_parameters(self, evaluations):
        self.inertia_weight = 0.9 - 0.7 * (evaluations / self.budget)
        self.quantum_jump_prob = max(0.1, 0.3 * (1 - self.gbest_fitness / (self.gbest_fitness + 1e-9)))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_swarm(func)
            evaluations += self.swarm_size

            if evaluations >= self.budget:
                break

            self.update_velocities_and_positions(lb, ub)
            self.quantum_jump(lb, ub)
            self.adapt_parameters(evaluations)

        return self.gbest_pos, self.gbest_fitness