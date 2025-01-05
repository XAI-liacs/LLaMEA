import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(10, min(50, budget // 10))
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight
        self.quantum_factor = 0.1

    def initialize_swarm(self, lb, ub):
        self.positions = lb + (ub - lb) * np.random.rand(self.swarm_size, self.dim)
        self.velocities = np.random.rand(self.swarm_size, self.dim) - 0.5
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))

    def evaluate_swarm(self, func):
        fitness = np.array([func(p) for p in self.positions])
        for i in range(self.swarm_size):
            if fitness[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = fitness[i]
                self.personal_best_positions[i] = self.positions[i]
            if fitness[i] < self.global_best_score:
                self.global_best_score = fitness[i]
                self.global_best_position = self.positions[i]

    def update_velocities_and_positions(self, lb, ub):
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
            quantum_velocity = self.quantum_factor * (np.random.rand(self.dim) - 0.5)
            self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component + quantum_velocity

            self.positions[i] += self.velocities[i]
            # Ensure the particles stay within bounds
            self.positions[i] = np.clip(self.positions[i], lb, ub)

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

        return self.global_best_position, self.global_best_score