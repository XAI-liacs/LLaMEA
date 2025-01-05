import numpy as np

class AdaptiveQuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
    
    def initialize_positions_and_velocities(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
    
    def update_velocity_and_position(self, index, func, evaluations):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_velocity = self.cognitive_coefficient * r1 * (self.personal_best_positions[index] - self.positions[index])
        social_velocity = self.social_coefficient * r2 * (self.global_best_position - self.positions[index])
        
        self.velocities[index] = (
            self.inertia_weight * self.velocities[index] +
            cognitive_velocity +
            social_velocity
        )
        
        self.positions[index] += self.velocities[index]
        self.positions[index] = np.clip(self.positions[index], func.bounds.lb, func.bounds.ub)
        
        current_score = func(self.positions[index])
        if current_score < self.personal_best_scores[index]:
            self.personal_best_scores[index] = current_score
            self.personal_best_positions[index] = np.copy(self.positions[index])
        
        return current_score
    
    def quantum_inspired_update(self, index, func):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        self.positions[index] += quantum_flip * (np.random.rand(self.dim) - 0.5)
        self.positions[index] = np.clip(self.positions[index], func.bounds.lb, func.bounds.ub)
        current_score = func(self.positions[index])
        
        if current_score < self.personal_best_scores[index]:
            self.personal_best_scores[index] = current_score
            self.personal_best_positions[index] = np.copy(self.positions[index])
        
        return current_score
    
    def __call__(self, func):
        self.initialize_positions_and_velocities(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                current_score = self.update_velocity_and_position(i, func, evaluations)
                evaluations += 1
                
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = np.copy(self.positions[i])
            
            if evaluations < self.budget:
                # Adjust inertia weight based on progress
                self.inertia_weight = self.inertia_weight_max - (evaluations / self.budget) * (self.inertia_weight_max - self.inertia_weight_min)
            
            # Quantum-inspired global search phase
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                current_score = self.quantum_inspired_update(i, func)
                evaluations += 1
                
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = np.copy(self.positions[i])

        return self.global_best_position