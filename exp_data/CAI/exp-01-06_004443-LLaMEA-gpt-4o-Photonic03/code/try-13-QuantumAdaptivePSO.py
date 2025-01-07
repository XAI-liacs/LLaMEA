import numpy as np

class QuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 + int(2 * np.sqrt(dim))  # Adaptive swarm size
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.c3 = 0.5  # Quantum-inspired attraction
        self.w = 0.9  # Initial inertia weight, changed from 0.7
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0
        self.local_best_positions = None  # New line: for local neighborhood strategy

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break
                score = func(self.positions[i])
                self.evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
            
            self.update_particles(lb, ub)
            self.w = 0.4 + 0.5 * (self.budget - self.evaluations) / self.budget  # Change: adapt inertia weight

        return self.global_best_position, self.global_best_score

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.local_best_positions = self.positions.copy()  # Initialize local best positions

    def update_particles(self, lb, ub):
        for i in range(self.swarm_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            q = np.random.rand(self.dim)

            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])
            quantum_velocity = self.c3 * q * (np.random.uniform(lb, ub, self.dim) - self.positions[i])
            
            local_best_velocity = self.c2 * r2 * (self.local_best_positions[i] - self.positions[i])  # New line: incorporate local best

            self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity + quantum_velocity + local_best_velocity
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)