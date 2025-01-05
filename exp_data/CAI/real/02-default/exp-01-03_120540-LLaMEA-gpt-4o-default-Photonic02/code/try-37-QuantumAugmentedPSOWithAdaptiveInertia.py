import numpy as np

class QuantumAugmentedPSOWithAdaptiveInertia:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.cognitive_const = 2.0
        self.social_const = 2.0
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
    
    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
    
    def update_positions_and_velocities(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_const * r1 * (self.best_positions[i] - self.positions[i])
            social_velocity = self.social_const * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity)
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)
    
    def quantum_position_update(self):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        quantum_adjustment = quantum_flip * (self.global_best_position - self.positions)
        self.positions += 0.05 * quantum_adjustment
    
    def __call__(self, func):
        self.initialize_particles(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                    
                current_score = func(self.positions[i])
                evaluations += 1
                
                if current_score < self.best_scores[i]:
                    self.best_scores[i] = current_score
                    self.best_positions[i] = self.positions[i]
                
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]
            
            self.update_positions_and_velocities(func)
            
            # Adaptive inertia weight
            self.inertia_weight = 0.9 - (0.8 * (evaluations / self.budget))
        
        # Final quantum-enhanced search
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            self.quantum_position_update()
            quantum_score = func(self.positions[i])
            evaluations += 1
            
            if quantum_score < self.global_best_score:
                self.global_best_score = quantum_score
                self.global_best_position = self.positions[i]