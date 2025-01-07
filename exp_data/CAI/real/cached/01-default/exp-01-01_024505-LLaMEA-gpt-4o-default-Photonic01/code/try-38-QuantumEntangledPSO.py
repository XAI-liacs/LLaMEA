import numpy as np

class QuantumEntangledPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(10, min(30, budget // 15))
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.global_best_position = None
        self.personal_best_scores = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.entanglement_factor = 0.3

    def initialize_particles(self, lb, ub):
        self.positions = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.rand(self.num_particles, self.dim) * (ub - lb) * 0.1
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))

    def update_velocities_and_positions(self, lb, ub):
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)

        cognitive_component = self.cognitive_constant * r1 * (self.personal_best_positions - self.positions)
        social_component = self.social_constant * r2 * (self.global_best_position - self.positions)
        
        self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
        self.positions += self.velocities
        
        self.positions = np.clip(self.positions, lb, ub)

    def apply_quantum_entanglement(self):
        for i in range(self.num_particles):
            if np.random.rand() < self.entanglement_factor:
                partner_idx = np.random.randint(self.num_particles)
                self.positions[i] = 0.5 * (self.positions[i] + self.positions[partner_idx]) + np.random.normal(0, 0.05, self.dim)

    def evaluate_particles(self, func):
        scores = np.array([func(pos) for pos in self.positions])
        for i in range(self.num_particles):
            if scores[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best_positions[i] = self.positions[i]
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.positions[i]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_particles(func)
            evaluations += self.num_particles
            
            if evaluations >= self.budget:
                break

            self.update_velocities_and_positions(lb, ub)
            self.apply_quantum_entanglement()

        return self.global_best_position, self.global_best_score