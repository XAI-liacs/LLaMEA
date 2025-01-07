import numpy as np

class HybridQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
    
    def quantum_inspired_perturbation(self, particle):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        perturbed_particle = particle + self.inertia_weight * quantum_flip
        return perturbed_particle
    
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
                    
                # Update velocities and positions
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coefficient * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_coefficient * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)
            
            # Quantum-inspired perturbation for global exploration
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                
                perturbed_position = self.quantum_inspired_perturbation(self.positions[i])
                perturbed_position = np.clip(perturbed_position, func.bounds.lb, func.bounds.ub)
                perturbed_score = func(perturbed_position)
                evaluations += 1
                
                if perturbed_score < self.global_best_score:
                    self.global_best_score = perturbed_score
                    self.global_best_position = perturbed_position