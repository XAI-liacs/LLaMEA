import numpy as np

class QuantumPSOWithAdaptiveVelocity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
    
    def update_velocity(self, idx):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[idx] - self.positions[idx])
        social_velocity = self.social_coeff * r2 * (self.best_position - self.positions[idx])
        quantum_tunneling = np.random.normal(0, 0.1, self.dim)
        return self.inertia_weight * self.velocities[idx] + cognitive_velocity + social_velocity + quantum_tunneling
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                current_score = func(self.positions[i])
                evaluations += 1
                
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = np.copy(self.positions[i])
                    
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = np.copy(self.positions[i])
                
                self.velocities[i] = self.update_velocity(i)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)
                
                # Adapt inertia weight based on progress
                self.inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
        
        # Perform quantum tunneling for the final set of optimizations
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            
            quantum_position = self.positions[i] + np.random.normal(0, 0.1, self.dim)
            quantum_position = np.clip(quantum_position, func.bounds.lb, func.bounds.ub)
            quantum_score = func(quantum_position)
            evaluations += 1
            
            if quantum_score < self.best_score:
                self.best_score = quantum_score
                self.best_position = quantum_position