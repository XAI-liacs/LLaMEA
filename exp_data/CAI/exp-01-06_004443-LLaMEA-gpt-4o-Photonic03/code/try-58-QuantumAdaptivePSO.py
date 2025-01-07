import numpy as np

class QuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 + int(2 * np.sqrt(dim))  # Adaptive swarm size
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.c3 = 0.5  # Quantum-inspired attraction
        self.w = 0.7  # Inertia weight
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

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
        
        return self.global_best_position, self.global_best_score

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        
    def update_particles(self, lb, ub):
        for i in range(self.swarm_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            q = np.random.rand(self.dim)

            cognitive_velocity = (self.c1 * (1 - self.evaluations/self.budget)) * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])
            quantum_velocity = (self.c3 * (1 - self.evaluations/self.budget)) * q * (np.random.uniform(lb, ub, self.dim) - self.positions[i])
            
            constriction_factor = 0.729 * (1 - self.evaluations/self.budget)  # Change made here for dynamic decay
            self.velocities[i] = constriction_factor * (((0.9 - 0.5 * self.evaluations/self.budget) * self.w) * self.velocities[i] + cognitive_velocity + social_velocity + quantum_velocity)
            self.velocities[i] = np.clip(self.velocities[i], -0.2*(ub-lb), 0.2*(ub-lb))  # Velocity clamping
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)