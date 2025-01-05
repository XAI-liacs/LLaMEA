import numpy as np

class DualLevelQuantumInspiredSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.best_individual_position = None
        self.best_individual_score = np.full(self.population_size, float('inf'))
        self.best_position = None
        self.best_score = float('inf')
        self.inertia_weight = 0.9
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.quantum_exploration_factor = 0.5
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
    
    def update_velocity_and_position(self, idx, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        
        # Update velocity with cognitive and social components
        cognitive_component = r1 * self.cognitive_coefficient * (self.best_individual_position[idx] - self.positions[idx])
        social_component = r2 * self.social_coefficient * (self.best_position - self.positions[idx])
        
        self.velocities[idx] = (
            self.inertia_weight * self.velocities[idx] +
            cognitive_component +
            social_component
        )
        
        # Update position
        self.positions[idx] += self.velocities[idx]
        self.positions[idx] = np.clip(self.positions[idx], lb, ub)
    
    def quantum_inspired_search(self, idx, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_flip = np.random.uniform(-self.quantum_exploration_factor, self.quantum_exploration_factor, self.dim)
        quantum_position = self.positions[idx] + quantum_flip
        quantum_position = np.clip(quantum_position, lb, ub)
        return quantum_position, func(quantum_position)
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                self.update_velocity_and_position(i, func)
                score = func(self.positions[i])
                evaluations += 1
                
                if score < self.best_individual_score[i]:
                    self.best_individual_score[i] = score
                    self.best_individual_position = self.positions[i].copy()
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.positions[i].copy()
            
            # Quantum-inspired global search
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                quantum_position, quantum_score = self.quantum_inspired_search(i, func)
                evaluations += 1
                
                if quantum_score < self.best_score:
                    self.best_score = quantum_score
                    self.best_position = quantum_position.copy()