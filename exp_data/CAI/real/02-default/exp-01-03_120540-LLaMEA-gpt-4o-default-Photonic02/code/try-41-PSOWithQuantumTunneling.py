import numpy as np

class PSOWithQuantumTunneling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
        self.personal_best_positions = None
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.quantum_tunneling_probability = 0.1
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
    
    def update_velocity(self, i):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_component = self.cognitive_coefficient * r1 * (self.personal_best_positions[i] - self.positions[i])
        social_component = self.social_coefficient * r2 * (self.best_position - self.positions[i])
        self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                              cognitive_component + social_component)
    
    def quantum_tunneling(self, position, bounds):
        if np.random.rand() < self.quantum_tunneling_probability:
            lb, ub = bounds.lb, bounds.ub
            # Quantum tunneling effect to escape local optima
            new_position = np.random.uniform(lb, ub, self.dim)
            return new_position
        return position
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                self.update_velocity(i)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)
                
                score = func(self.positions[i])
                evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.positions[i]
            
            # Quantum tunneling step for all particles
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                tunneled_position = self.quantum_tunneling(self.positions[i], func.bounds)
                tunneled_score = func(tunneled_position)
                evaluations += 1
                
                if tunneled_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = tunneled_score
                    self.personal_best_positions[i] = tunneled_position
                
                if tunneled_score < self.best_score:
                    self.best_score = tunneled_score
                    self.best_position = tunneled_position