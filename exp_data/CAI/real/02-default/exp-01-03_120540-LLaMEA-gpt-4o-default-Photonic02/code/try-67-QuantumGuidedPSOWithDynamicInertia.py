import numpy as np

class QuantumGuidedPSOWithDynamicInertia:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.initial_inertia_weight = self.inertia_weight
    
    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
    
    def quantum_guided_update(self, position):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        return position + 0.1 * quantum_flip
    
    def update_velocities_and_positions(self, func, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            inertia_velocity = self.inertia_weight * self.velocities[i]
            cognitive_component = self.cognitive_coefficient * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.social_coefficient * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
            
            self.velocities[i] = inertia_velocity + cognitive_component + social_component
            self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)
            
            current_score = func(self.positions[i])
            evaluations += 1
            
            if current_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = current_score
                self.personal_best_positions[i] = self.positions[i]
            
            if current_score < self.global_best_score:
                self.global_best_score = current_score
                self.global_best_position = self.positions[i]
    
    def __call__(self, func):
        self.initialize_particles(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            self.update_velocities_and_positions(func, evaluations)
            
            # Update inertia weight dynamically
            self.inertia_weight = self.initial_inertia_weight * (1 - evaluations / self.budget)
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Quantum-guided position update
                quantum_position = self.quantum_guided_update(self.positions[i])
                quantum_position = np.clip(quantum_position, func.bounds.lb, func.bounds.ub)
                quantum_score = func(quantum_position)
                evaluations += 1
                
                if quantum_score < self.global_best_score:
                    self.global_best_score = quantum_score
                    self.global_best_position = quantum_position