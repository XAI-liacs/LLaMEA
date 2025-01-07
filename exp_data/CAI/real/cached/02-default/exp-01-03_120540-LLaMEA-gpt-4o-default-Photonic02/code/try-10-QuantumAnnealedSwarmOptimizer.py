import numpy as np

class QuantumAnnealedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.temperature = 1.0
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
    
    def update_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.clip(self.positions, lb, ub)
    
    def quantum_tunneling(self, position):
        tunneling_probability = np.exp(-self.global_best_score / self.temperature)
        if np.random.rand() < tunneling_probability:
            return position + np.random.normal(0, 0.1, self.dim)
        return position
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.social_coeff * r2 * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] + 
                                      cognitive_component + 
                                      social_component)
                
                self.positions[i] += self.velocities[i]
                self.update_positions(func.bounds)
                
                # Apply quantum tunneling
                new_position = self.quantum_tunneling(self.positions[i])
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                new_score = func(new_position)
                evaluations += 1
                
                if new_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = new_score
                    self.personal_best_positions[i] = new_position
                
                if new_score < self.global_best_score:
                    self.global_best_score = new_score
                    self.global_best_position = new_position
                
                # Cooling schedule for simulated annealing
                self.temperature *= 0.99