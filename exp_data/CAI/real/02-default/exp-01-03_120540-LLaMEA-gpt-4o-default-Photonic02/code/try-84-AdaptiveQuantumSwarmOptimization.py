import numpy as np

class AdaptiveQuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
        self.local_best_positions = None
        self.local_best_scores = None
        self.global_influence = 0.5
        self.personal_influence = 0.5
        self.quantum_influence = 0.3
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.local_best_positions = np.copy(self.positions)
        self.local_best_scores = np.full(self.swarm_size, float('inf'))
    
    def quantum_perturbation(self, position):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        perturbed_position = position + self.quantum_influence * quantum_flip
        return perturbed_position
    
    def update_velocity_and_position(self):
        r1 = np.random.rand(self.swarm_size, self.dim)
        r2 = np.random.rand(self.swarm_size, self.dim)
        
        cognitive_component = self.personal_influence * r1 * (self.local_best_positions - self.positions)
        social_component = self.global_influence * r2 * (self.best_position - self.positions)
        
        self.velocities += cognitive_component + social_component
        self.positions += self.velocities
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                
                current_position = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)
                current_score = func(current_position)
                evaluations += 1
                
                if current_score < self.local_best_scores[i]:
                    self.local_best_scores[i] = current_score
                    self.local_best_positions[i] = current_position
                
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = current_position
            
            self.update_velocity_and_position()
            
            # Quantum-inspired search for diversity enhancement
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                
                perturbed_position = self.quantum_perturbation(self.positions[i])
                perturbed_position = np.clip(perturbed_position, func.bounds.lb, func.bounds.ub)
                perturbed_score = func(perturbed_position)
                evaluations += 1
                
                if perturbed_score < self.local_best_scores[i]:
                    self.local_best_scores[i] = perturbed_score
                    self.local_best_positions[i] = perturbed_position
                
                if perturbed_score < self.best_score:
                    self.best_score = perturbed_score
                    self.best_position = perturbed_position