import numpy as np

class QuantumInspiredSGD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.learning_rate = 0.1
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
    
    def quantum_inspired_stochastic_update(self, individual, gradient):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        stochastic_gradient = gradient * quantum_flip
        return individual - self.learning_rate * stochastic_gradient
    
    def compute_gradient_estimate(self, func, position):
        epsilon = 1e-8
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturbed_position = np.copy(position)
            perturbed_position[i] += epsilon
            gradient[i] = (func(perturbed_position) - func(position)) / epsilon
        return gradient
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                current_position = self.positions[i]
                gradient_estimate = self.compute_gradient_estimate(func, current_position)
                
                updated_position = self.quantum_inspired_stochastic_update(current_position, gradient_estimate)
                updated_position = np.clip(updated_position, func.bounds.lb, func.bounds.ub)
                updated_score = func(updated_position)
                evaluations += 1
                
                if updated_score < self.best_score:
                    self.best_score = updated_score
                    self.best_position = updated_position
                
                if updated_score < func(current_position):
                    self.positions[i] = updated_position
        
        return self.best_position, self.best_score