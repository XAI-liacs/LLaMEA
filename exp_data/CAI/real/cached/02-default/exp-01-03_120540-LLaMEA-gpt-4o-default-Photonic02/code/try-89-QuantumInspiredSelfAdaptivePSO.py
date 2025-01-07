import numpy as np

class QuantumInspiredSelfAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))

    def quantum_inspired_velocity_update(self, velocity, position, global_best):
        # Quantum-inspired influence on velocity
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        return velocity + quantum_flip * (global_best - position)
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Evaluate particle fitness
                score = func(self.positions[i])
                evaluations += 1
                
                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                    
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
                
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (w * self.velocities[i] + cognitive_component + social_component)
                
                # Quantum-inspired velocity adjustment
                self.velocities[i] = self.quantum_inspired_velocity_update(self.velocities[i], self.positions[i], self.global_best_position)
                
                # Update position
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)
        
        # Return best solution found
        return self.global_best_position, self.global_best_score