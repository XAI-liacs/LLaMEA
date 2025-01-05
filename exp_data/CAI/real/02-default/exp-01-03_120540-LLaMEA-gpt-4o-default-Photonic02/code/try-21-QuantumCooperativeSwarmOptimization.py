import numpy as np

class QuantumCooperativeSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.position = None
        self.velocity = None
        self.best_personal_position = None
        self.best_global_position = None
        self.best_score = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.position = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.best_personal_position = np.copy(self.position)
        self.best_global_position = np.copy(self.position[0])
    
    def quantum_flip(self, position, bounds):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        flipped_position = position + 0.1 * quantum_flip
        return np.clip(flipped_position, bounds.lb, bounds.ub)
    
    def update_velocity_position(self, particle_idx):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_velocity = self.cognitive_coeff * r1 * (self.best_personal_position[particle_idx] - self.position[particle_idx])
        social_velocity = self.social_coeff * r2 * (self.best_global_position - self.position[particle_idx])
        
        self.velocity[particle_idx] = self.inertia_weight * self.velocity[particle_idx] + cognitive_velocity + social_velocity
        self.position[particle_idx] += self.velocity[particle_idx]
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                
                self.position[i] = np.clip(self.position[i], func.bounds.lb, func.bounds.ub)
                score = func(self.position[i])
                evaluations += 1
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_global_position = self.position[i]
                
                if score < func(self.best_personal_position[i]):
                    self.best_personal_position[i] = self.position[i]
                
                self.update_velocity_position(i)
            
            # Quantum-inspired cooperative behavior
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                    
                quantum_position = self.quantum_flip(self.position[i], func.bounds)
                quantum_score = func(quantum_position)
                evaluations += 1
                
                if quantum_score < self.best_score:
                    self.best_score = quantum_score
                    self.best_global_position = quantum_position