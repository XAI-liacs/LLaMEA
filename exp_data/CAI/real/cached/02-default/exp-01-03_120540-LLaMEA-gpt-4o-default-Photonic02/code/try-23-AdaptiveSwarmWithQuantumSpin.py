import numpy as np

class AdaptiveSwarmWithQuantumSpin:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.personal_best_positions = None
        self.global_best_position = None
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_score = float('inf')
        self.velocities = None
        self.positions = None
        self.inertia_weight = 0.9
        self.cognitive_constant = 2.0
        self.social_constant = 2.0
        self.quantum_spin_constant = 0.05
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
    
    def update_velocities_and_positions(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_constant * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.social_constant * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
            
            # Quantum spin-inspired update
            spin_flip_probability = np.random.rand(self.dim) < self.quantum_spin_constant
            self.velocities[i] += spin_flip_probability * np.random.uniform(-1, 1, self.dim)
            
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                
                score = func(self.positions[i])
                evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
            
            self.update_velocities_and_positions(func)
            
            # Adapt inertia weight
            self.inertia_weight = 0.4 + 0.5 * (self.budget - evaluations) / self.budget